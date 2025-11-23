"""
AWS Bedrock client wrapper for interacting with Claude Sonnet 4.5 model.
Provides methods for invoking the model with and without streaming.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Iterator, Union, Callable
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from botocore.config import Config

# Configure logging first
logger = logging.getLogger(__name__)

# Try to import tiktoken, but make it optional
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, token counting will use approximation")


class BedrockError(Exception):
    """Base exception for Bedrock client errors."""
    
    def __init__(self, message: str, error_type: str = 'general', original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class BedrockClient:
    """
    Wrapper class for AWS Bedrock runtime client.
    Handles model invocation, streaming, and error handling with retry logic.
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0  # seconds
    MAX_BACKOFF = 32.0  # seconds
    BACKOFF_MULTIPLIER = 2.0
    
    # Retryable error codes
    RETRYABLE_ERRORS = {
        'ThrottlingException',
        'ServiceUnavailableException',
        'InternalServerException',
        'ModelTimeoutException',
        'TooManyRequestsException'
    }
    
    def __init__(self, credentials: Dict[str, str], model_id: str, max_tokens: int = 2048, temperature: float = 0.7):
        """
        Initialize Bedrock client with AWS credentials.
        
        Args:
            credentials: Dictionary containing AWS credentials
                - aws_access_key_id: AWS access key
                - aws_secret_access_key: AWS secret key
                - region_name: AWS region
            model_id: Bedrock model identifier (e.g., 'anthropic.claude-sonnet-4-5-v2')
            max_tokens: Maximum tokens for model response
            temperature: Model temperature for response generation
        
        Raises:
            BedrockError: If client initialization fails
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize tokenizer for token counting if available
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("Token counting enabled with tiktoken")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer: {e}, token counting will be approximate")
        else:
            logger.info("Token counting will use approximation (tiktoken not available)")
        
        try:
            # Configure boto3 with timeout and retry settings
            config = Config(
                read_timeout=300,
                connect_timeout=60,
                retries={'max_attempts': 0},  # We'll handle retries manually
                max_pool_connections=50  # Connection pooling for better performance
            )
            
            self.client = boto3.client(
                service_name='bedrock-runtime',
                aws_access_key_id=credentials['aws_access_key_id'],
                aws_secret_access_key=credentials['aws_secret_access_key'],
                region_name=credentials['region_name'],
                config=config
            )
            
            logger.info(f"Bedrock client initialized successfully for model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise BedrockError(
                f"Failed to initialize Bedrock client: {str(e)}",
                error_type='initialization',
                original_error=e
            )
    
    def _retry_with_backoff(self, operation: Callable, operation_name: str) -> Any:
        """
        Execute an operation with exponential backoff retry logic.
        
        Args:
            operation: Callable to execute
            operation_name: Name of the operation for logging
        
        Returns:
            Result from the operation
        
        Raises:
            BedrockError: If all retry attempts fail
        """
        last_error = None
        backoff = self.INITIAL_BACKOFF
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return operation()
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                # Check if error is retryable
                if error_code in self.RETRYABLE_ERRORS and attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"{operation_name} attempt {attempt + 1} failed with {error_code}. "
                        f"Retrying in {backoff:.2f} seconds..."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
                    last_error = e
                    continue
                
                # Non-retryable error or max retries reached
                if error_code in ['InvalidSignatureException', 'UnrecognizedClientException', 'AccessDeniedException']:
                    error_type = 'auth'
                    logger.error(f"Authentication error in {operation_name}: {error_message}")
                elif error_code in self.RETRYABLE_ERRORS:
                    error_type = 'service'
                    logger.error(f"Service error in {operation_name} after {attempt + 1} attempts: {error_message}")
                else:
                    error_type = 'model'
                    logger.error(f"Model error in {operation_name}: {error_message}")
                
                raise BedrockError(
                    f"{operation_name} failed: {error_message}",
                    error_type=error_type,
                    original_error=e
                )
            
            except (EndpointConnectionError, ReadTimeoutError) as e:
                # Network errors are retryable
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"{operation_name} attempt {attempt + 1} failed with network error. "
                        f"Retrying in {backoff:.2f} seconds..."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
                    last_error = e
                    continue
                
                logger.error(f"Network error in {operation_name} after {attempt + 1} attempts")
                raise BedrockError(
                    "Network connection error. Please check your internet connection.",
                    error_type='network',
                    original_error=e
                )
            
            except Exception as e:
                # Unexpected errors are not retried
                logger.error(f"Unexpected error in {operation_name}: {str(e)}")
                raise BedrockError(
                    f"Unexpected error during {operation_name}: {str(e)}",
                    error_type='general',
                    original_error=e
                )
        
        # Should not reach here, but just in case
        if last_error:
            raise BedrockError(
                f"{operation_name} failed after {self.MAX_RETRIES} attempts",
                error_type='service',
                original_error=last_error
            )
    
    def invoke_model(self, prompt: str, context: Optional[str] = None, chat_history: Optional[list] = None, query_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a prompt to Claude Sonnet 4.5 and get a response with automatic retry.
        
        Args:
            prompt: User's input message
            context: Optional context from knowledge base retrieval
            chat_history: Optional list of previous messages for conversation context
            query_type: Optional query classification (e.g., 'counting', 'aggregation', 'general')
        
        Returns:
            Dictionary containing:
                - text: Model response text
                - metadata: Additional response metadata (tokens, latency, etc.)
        
        Raises:
            BedrockError: If model invocation fails after retries
        """
        # Format the request payload
        request_body = self._format_request(prompt, context, chat_history, query_type)
        
        # Track latency
        start_time = time.time()
        
        # Define the operation to retry
        def _invoke():
            return self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
        
        # Execute with retry logic
        response = self._retry_with_backoff(_invoke, "Model invocation")
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Parse the response
        result = self._parse_response(response)
        result['metadata']['latency_ms'] = latency_ms
        
        logger.info(f"Model invocation successful. Latency: {latency_ms}ms")
        
        return result

    def invoke_model_with_streaming(self, prompt: str, context: Optional[str] = None, chat_history: Optional[list] = None, query_type: Optional[str] = None) -> Iterator[str]:
        """
        Send a prompt to Claude Sonnet 4.5 and stream the response with automatic retry.
        
        Args:
            prompt: User's input message
            context: Optional context from knowledge base retrieval
            chat_history: Optional list of previous messages for conversation context
            query_type: Optional query classification (e.g., 'counting', 'aggregation', 'general')
        
        Yields:
            Chunks of text as they are received from the model
        
        Raises:
            BedrockError: If model invocation fails after retries
        """
        # Format the request payload
        request_body = self._format_request(prompt, context, chat_history, query_type)
        
        # Define the operation to retry (only the initial connection)
        def _invoke_stream():
            return self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
        
        # Execute with retry logic
        try:
            response = self._retry_with_backoff(_invoke_stream, "Streaming invocation")
            
            # Stream the response chunks
            stream = response.get('body')
            if stream:
                logger.info("Streaming response started")
                for event in stream:
                    chunk = event.get('chunk')
                    if chunk:
                        try:
                            chunk_data = json.loads(chunk.get('bytes').decode())
                            
                            # Extract text from chunk based on response format
                            if 'delta' in chunk_data:
                                # Streaming format with delta
                                delta = chunk_data['delta']
                                if 'text' in delta:
                                    yield delta['text']
                            elif 'completion' in chunk_data:
                                # Alternative format
                                yield chunk_data['completion']
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode chunk: {str(e)}")
                            continue
                
                logger.info("Streaming response completed")
        
        except BedrockError:
            # Re-raise BedrockError from retry logic
            raise
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {str(e)}")
            raise BedrockError(
                f"Unexpected error during streaming invocation: {str(e)}",
                error_type='general',
                original_error=e
            )
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Approximate token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback: approximate 1 token per 4 characters
        return len(text) // 4
    
    def _format_request(self, prompt: str, context: Optional[str] = None, chat_history: Optional[list] = None, query_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Format the request payload for Bedrock API with token limit checking.
        
        Args:
            prompt: User's input message
            context: Optional context from knowledge base
            chat_history: Optional conversation history
            query_type: Optional query classification (e.g., 'counting', 'aggregation', 'general')
        
        Returns:
            Formatted request body for Bedrock API
        
        Raises:
            BedrockError: If token count exceeds model limits
        """
        # Build the system prompt with structured data formatting instructions
        system_prompt = """You are a helpful AI assistant specialized in analyzing CSV data and metadata from warehouse management systems.

CORE CAPABILITIES:
You can work with both actual CSV data and metadata about tables. When metadata is available but actual data is not, you can still provide valuable insights about table structure, relationships, and what queries would be needed.

DATA SOURCES:
The knowledge base contains:
1. CSV files with actual tabular data (t_employee.csv, t_location.csv, t_inventory.csv, t_po_detail.csv, etc.)
2. Metadata files (optimized_metadata_full_v10.json) with table schemas, column types, relationships, and statistics

ANSWERING STRATEGIES:

For COUNTING/AGGREGATION queries ("how many", "count", "total", "sum"):
1. First, check if actual CSV data is in the context - if yes, count the rows/values directly
2. If only metadata is available:
   - Look for row count statistics in the metadata
   - Check for the column being counted and explain its purpose
   - Search for related tables that might have the data through foreign keys
3. If neither data nor metadata is available:
   - Explain which table would contain this information
   - Suggest what needs to be indexed in the knowledge base
   - Look for indirect evidence in related tables (e.g., employee_id in transaction logs)

For GENERAL queries about data:
1. Use actual CSV data when available to provide specific answers
2. Use metadata to understand table structure and relationships
3. Combine both sources for comprehensive answers

HANDLING PARTIAL/MISSING DATA:
- Be transparent about data availability
- When actual data is missing but metadata exists, say: "Based on the table metadata, [table_name] contains [columns]. However, the actual data is not currently indexed. To answer this precisely, the CSV file would need to be added to the knowledge base."
- When you find related data in other tables, mention it: "While [primary_table] is not available, I found [related_info] in [other_table]"
- For employee queries specifically: Check t_tran_log, t_holds, t_exception_log, t_audit_log for employee_id references

OUTPUT FORMAT:
When data is available:
- Present in markdown tables with clear headers
- Include a "Summary:" section with key insights
- Use bullet points for clarity

When data is not available:
- Explain what table would contain the answer
- Describe the table structure from metadata
- Suggest what needs to be indexed
- Provide any indirect evidence from related tables

Example with data:
| Employee ID | Name        | Department |
|-------------|-------------|------------|
| E001        | John Smith  | Warehouse  |
| E002        | Jane Doe    | Logistics  |

Summary:
- 2 employees found
- Departments: Warehouse (1), Logistics (1)

Example without data:
Based on the metadata, t_employee.csv contains employee information with columns: employee_id, name, department, hire_date. However, the actual employee data is not currently indexed in the knowledge base. To get an accurate count, this file would need to be added. I can see employee_id references in the transaction logs (E001, E002, E005) which suggests at least 3 employees exist in the system."""

        # Add query-specific instructions based on classification
        if query_type == 'counting':
            system_prompt += """

SPECIAL INSTRUCTIONS FOR THIS COUNTING QUERY:
- Your primary goal is to provide an accurate count
- If actual data is available, count the rows/items directly and show your work
- If only metadata is available, look for row count statistics or cardinality information
- If neither is available, be explicit: "I cannot provide an exact count because [reason]"
- Always look for indirect evidence in related tables through foreign key relationships
- For employee counts: Check all tables with employee_id columns for unique values"""
        
        elif query_type == 'aggregation':
            system_prompt += """

SPECIAL INSTRUCTIONS FOR THIS AGGREGATION QUERY:
- Focus on calculating sums, averages, or other aggregate values
- If actual data is available, perform the calculation and show the formula
- If only metadata is available, explain what calculation would be performed
- Look for summary statistics in metadata that might answer the question
- Be clear about any assumptions made in your calculations"""

        if context:
            system_prompt += f"\n\nCONTEXT FROM KNOWLEDGE BASE:\n{context}"
        else:
            system_prompt += "\n\nNOTE: No context was retrieved from the knowledge base for this query. Provide a helpful response based on general knowledge about the data structure, or explain what information would be needed."
        
        # Build messages array
        messages = []
        
        # Add chat history if available (limit to last 10 messages)
        if chat_history:
            history_to_include = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in history_to_include:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Count tokens to prevent exceeding limits
        total_tokens = self.count_tokens(system_prompt if context else "")
        for msg in messages:
            total_tokens += self.count_tokens(msg.get("content", ""))
        
        # Claude models typically have 200k context window, but we'll be conservative
        # Reserve space for response (max_tokens) and some buffer
        max_input_tokens = 180000 - self.max_tokens
        
        if total_tokens > max_input_tokens:
            logger.warning(f"Token count ({total_tokens}) exceeds limit ({max_input_tokens}), truncating history")
            
            # Truncate history to fit within limits
            while len(messages) > 1 and total_tokens > max_input_tokens:
                # Remove oldest message (but keep the current prompt)
                removed_msg = messages.pop(0)
                total_tokens -= self.count_tokens(removed_msg.get("content", ""))
            
            # If still too large, truncate context
            if total_tokens > max_input_tokens and context:
                logger.warning("Truncating context to fit token limits")
                context_tokens = self.count_tokens(context)
                reduction_needed = total_tokens - max_input_tokens
                
                if reduction_needed < context_tokens:
                    # Truncate context proportionally
                    keep_ratio = (context_tokens - reduction_needed) / context_tokens
                    context = context[:int(len(context) * keep_ratio)]
                    system_prompt = f"You are a helpful AI assistant.\n\nUse the following context to answer the user's question:\n{context}"
                    total_tokens = max_input_tokens
                else:
                    # Remove context entirely
                    context = None
                    system_prompt = "You are a helpful AI assistant."
                    total_tokens -= context_tokens
        
        logger.debug(f"Request token count: ~{total_tokens}")
        
        # Format request according to Claude API specification
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        # Add system prompt if we have context
        if context:
            request_body["system"] = system_prompt
        
        return request_body
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response from Bedrock API and extract text.
        
        Args:
            response: Raw response from Bedrock API
        
        Returns:
            Dictionary containing:
                - text: Extracted response text
                - metadata: Response metadata (model_id, tokens, etc.)
        
        Raises:
            BedrockError: If response parsing fails
        """
        try:
            # Read and parse response body
            response_body = json.loads(response.get('body').read())
            
            # Extract text from response
            text = ""
            if 'content' in response_body:
                # Claude format with content array
                for content_block in response_body['content']:
                    if content_block.get('type') == 'text':
                        text += content_block.get('text', '')
            elif 'completion' in response_body:
                # Alternative format
                text = response_body['completion']
            else:
                raise BedrockError(
                    "Unable to extract text from model response",
                    error_type='parsing'
                )
            
            # Extract metadata
            metadata = {
                'model_id': self.model_id,
                'stop_reason': response_body.get('stop_reason', 'unknown'),
                'usage': response_body.get('usage', {})
            }
            
            return {
                'text': text,
                'metadata': metadata
            }
            
        except json.JSONDecodeError as e:
            raise BedrockError(
                f"Failed to parse JSON response: {str(e)}",
                error_type='parsing',
                original_error=e
            )
        except Exception as e:
            raise BedrockError(
                f"Failed to parse response: {str(e)}",
                error_type='parsing',
                original_error=e
            )
