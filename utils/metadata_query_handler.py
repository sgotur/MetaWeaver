"""
Metadata query handler module for processing natural language queries against pandas DataFrames.
Orchestrates query generation, execution, and summarization using LLM.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from utils.aws_client import BedrockClient, BedrockError
from utils.csv_data_loader import CSVDataLoader, CSVLoaderError

# Configure logging
logger = logging.getLogger(__name__)


class MetadataQueryError(Exception):
    """Base exception for metadata query errors."""
    
    def __init__(self, message: str, error_type: str = 'general', original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class MetadataQueryHandler:
    """
    Handles natural language queries against pandas DataFrames.
    Converts user queries to pandas expressions, executes them safely, and generates summaries.
    """
    
    # Query timeout in seconds
    QUERY_TIMEOUT = 30
    
    # Maximum rows to return in results
    MAX_RESULT_ROWS = 10000
    
    # Unsafe operations that should be blocked
    UNSAFE_OPERATIONS = [
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'execfile', 'reload', 'vars', 'globals',
        'locals', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
    ]
    
    def __init__(
        self, 
        bedrock_client: BedrockClient, 
        metadata_df: pd.DataFrame,
        csv_loader: Optional[CSVDataLoader] = None
    ):
        """
        Initialize MetadataQueryHandler with required dependencies.
        
        Args:
            bedrock_client: BedrockClient instance for LLM invocation
            metadata_df: pandas DataFrame containing the metadata catalog
            csv_loader: Optional CSVDataLoader for loading actual data files
        
        Raises:
            MetadataQueryError: If initialization fails
        """
        if not isinstance(metadata_df, pd.DataFrame):
            raise MetadataQueryError(
                "metadata_df must be a pandas DataFrame",
                error_type='initialization'
            )
        
        if metadata_df.empty:
            raise MetadataQueryError(
                "metadata_df cannot be empty",
                error_type='initialization'
            )
        
        self.bedrock_client = bedrock_client
        self.metadata_df = metadata_df
        self.csv_loader = csv_loader
        
        # Extract schema information for query generation
        self.df_schema = self._extract_schema(metadata_df)
        
        # Cache for loaded CSV DataFrames
        self.csv_cache = {}
        
        logger.info(f"MetadataQueryHandler initialized with metadata catalog shape: {metadata_df.shape}")
        if csv_loader:
            logger.info("CSV data loader enabled for querying actual data files")
    
    def _identify_target_table(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Identify which table/CSV file the user is asking about using LLM.
        
        Args:
            user_query: User's natural language query
        
        Returns:
            Dictionary with table information or None if querying metadata catalog
            {
                'table_name': str,
                'file': str,
                'description': str,
                'columns': list
            }
        """
        try:
            # Build prompt for table identification
            prompt = f"""You are a data analyst. Analyze the user's question and determine which table they want to query.

Available Tables:
"""
            
            # Add table information from metadata
            for idx, row in self.metadata_df.iterrows():
                table_name = row.get('name', 'Unknown')
                description = row.get('description', 'No description')
                file = row.get('file', '')
                
                prompt += f"\n{idx + 1}. {table_name}"
                if file:
                    prompt += f" (File: {file})"
                prompt += f"\n   Description: {description}\n"
            
            prompt += f"""
User Question: {user_query}

Analyze the question and determine:
1. Is the user asking about the actual data IN a table (e.g., "show me employees", "list products")?
2. Or are they asking about the metadata/structure (e.g., "what tables exist", "describe the schema")?

If asking about actual data:
- Respond with ONLY the table name (e.g., "t_employee")
- Choose the most relevant table based on the question

If asking about metadata/structure:
- Respond with: "METADATA"

Response (table name or METADATA):"""
            
            # Invoke LLM
            response = self.bedrock_client.invoke_model(
                prompt=prompt,
                context=None,
                chat_history=None
            )
            
            table_identifier = response.get('text', '').strip().strip('"\'')
            
            logger.info(f"Table identification result: {table_identifier}")
            
            # Check if user wants metadata
            if table_identifier.upper() == 'METADATA':
                logger.info("User query is about metadata catalog, not actual data")
                return None
            
            # Find the table in metadata
            matching_tables = self.metadata_df[
                self.metadata_df['name'].str.lower() == table_identifier.lower()
            ]
            
            if len(matching_tables) == 0:
                logger.warning(f"Table '{table_identifier}' not found in metadata catalog")
                return None
            
            table_row = matching_tables.iloc[0]
            
            return {
                'table_name': table_row.get('name'),
                'file': table_row.get('file'),
                'description': table_row.get('description'),
                'columns': table_row.get('columns', [])
            }
            
        except Exception as e:
            logger.error(f"Table identification failed: {str(e)}")
            # Fall back to metadata query
            return None
    
    def process_query(
        self, 
        user_query: str, 
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration pipeline for processing metadata queries.
        
        This method:
        1. Generates pandas query expression from natural language
        2. Validates and executes query safely
        3. Generates natural language summary of results
        4. Returns structured response
        
        Args:
            user_query: Natural language query from user
            chat_history: Optional conversation history for context
        
        Returns:
            Dictionary containing:
                - success: bool indicating if query succeeded
                - results: pd.DataFrame with query results (if successful)
                - summary: str with natural language summary
                - query_used: str with pandas expression used
                - row_count: int number of rows returned
                - error: Optional[str] error message if failed
                - timestamp: str ISO format timestamp
        """
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Step 0: Identify target table (if CSV loader is available)
            target_table = None
            target_df = self.metadata_df
            data_source = "metadata catalog"
            
            if self.csv_loader:
                target_table = self._identify_target_table(user_query)
                
                if target_table:
                    # User wants to query actual data
                    table_name = target_table['table_name']
                    csv_file = target_table['file']
                    
                    logger.info(f"User query targets table: {table_name} (file: {csv_file})")
                    
                    # Check cache first
                    if csv_file in self.csv_cache:
                        logger.info(f"Using cached DataFrame for {csv_file}")
                        target_df = self.csv_cache[csv_file]
                    else:
                        # Load CSV data
                        logger.info(f"Loading CSV data: {csv_file}")
                        csv_result = self.csv_loader.load_csv_data(csv_file)
                        
                        if not csv_result['success']:
                            return {
                                'success': False,
                                'results': None,
                                'summary': '',
                                'query_used': '',
                                'row_count': 0,
                                'error': f"Failed to load data file: {csv_result['error']}",
                                'timestamp': datetime.now().isoformat(),
                                'data_source': None
                            }
                        
                        target_df = csv_result['dataframe']
                        self.csv_cache[csv_file] = target_df
                        logger.info(f"Loaded {len(target_df)} rows from {csv_file}")
                    
                    data_source = f"{table_name} ({csv_file})"
            
            # Step 1: Generate pandas query expression
            try:
                pandas_query = self._generate_pandas_query(user_query, chat_history, target_df)
                logger.info(f"Generated query: {pandas_query}")
                print(f"\n{'='*60}")
                print(f"GENERATED PANDAS QUERY:")
                print(f"{'='*60}")
                print(f"{pandas_query}")
                print(f"Data Source: {data_source}")
                print(f"{'='*60}\n")
            except Exception as e:
                logger.error(f"Query generation failed: {str(e)}")
                return {
                    'success': False,
                    'results': None,
                    'summary': '',
                    'query_used': '',
                    'row_count': 0,
                    'error': f"Failed to generate query: {str(e)}",
                    'timestamp': datetime.now().isoformat(),
                    'data_source': data_source
                }
            
            # Step 2: Execute query safely
            try:
                results_df = self._execute_query_safely(pandas_query, target_df)
                row_count = len(results_df)
                logger.info(f"Query executed successfully, returned {row_count} rows")
            except Exception as e:
                logger.error(f"Query execution failed: {str(e)}")
                return {
                    'success': False,
                    'results': None,
                    'summary': '',
                    'query_used': pandas_query,
                    'row_count': 0,
                    'error': f"Failed to execute query: {str(e)}",
                    'timestamp': datetime.now().isoformat(),
                    'data_source': data_source
                }
            
            # Step 3: Generate summary
            try:
                summary = self._generate_summary(user_query, results_df, pandas_query, data_source)
                logger.info("Summary generated successfully")
            except Exception as e:
                logger.warning(f"Summary generation failed: {str(e)}, returning results without summary")
                summary = f"Query returned {row_count} rows from {data_source}. (Summary generation failed)"
            
            # Step 4: Return structured response
            return {
                'success': True,
                'results': results_df,
                'summary': summary,
                'query_used': pandas_query,
                'row_count': row_count,
                'error': None,
                'timestamp': datetime.now().isoformat(),
                'data_source': data_source
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in process_query: {str(e)}")
            return {
                'success': False,
                'results': None,
                'summary': '',
                'query_used': '',
                'row_count': 0,
                'error': f"Unexpected error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_pandas_query(
        self, 
        user_query: str, 
        chat_history: Optional[List[Dict]] = None,
        target_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate pandas query expression from natural language using LLM.
        
        Args:
            user_query: Natural language query from user
            chat_history: Optional conversation history for context
            target_df: Optional target DataFrame (if None, uses metadata_df)
        
        Returns:
            String containing pandas query expression
        
        Raises:
            MetadataQueryError: If query generation fails
        """
        # Use target DataFrame or fall back to metadata
        df_to_query = target_df if target_df is not None else self.metadata_df
        
        # Build prompt for query generation
        prompt = self._build_query_generation_prompt(user_query, df_to_query)
        
        try:
            # Invoke LLM to generate query
            response = self.bedrock_client.invoke_model(
                prompt=prompt,
                context=None,
                chat_history=chat_history
            )
            
            query_text = response.get('text', '').strip()
            
            # Extract query from response (remove markdown code blocks if present)
            query = self._extract_query_from_response(query_text)
            
            # Validate query before returning
            self._validate_query(query)
            
            return query
            
        except BedrockError as e:
            raise MetadataQueryError(
                f"LLM invocation failed: {str(e)}",
                error_type='llm_error',
                original_error=e
            )
        except Exception as e:
            raise MetadataQueryError(
                f"Query generation failed: {str(e)}",
                error_type='generation_error',
                original_error=e
            )
    
    def _execute_query_safely(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute pandas query in a sandboxed manner with safety checks.
        
        Args:
            query: Pandas query expression to execute
            df: DataFrame to query against
        
        Returns:
            DataFrame containing query results
        
        Raises:
            MetadataQueryError: If query execution fails or is unsafe
        """
        # Final safety check before execution
        self._validate_query(query)
        
        try:
            # Create a restricted namespace for query execution
            # Allow safe built-in functions that pandas operations might need
            safe_builtins = {
                'isinstance': isinstance,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'any': any,
                'all': all,
                'type': type,
            }
            
            namespace = {
                'df': df,
                'pd': pd,
                '__builtins__': safe_builtins
            }
            
            # Execute query in restricted namespace
            result = eval(query, namespace)
            
            # Convert result to DataFrame if needed
            if not isinstance(result, pd.DataFrame):
                if isinstance(result, pd.Series):
                    # Series to DataFrame
                    result = result.to_frame()
                elif isinstance(result, (int, float, str, bool)) or hasattr(result, 'item'):
                    # Scalar value (including numpy scalars like int64, float64) to DataFrame
                    # numpy scalars have .item() method to convert to Python scalar
                    scalar_val = result.item() if hasattr(result, 'item') else result
                    result = pd.DataFrame({'result': [scalar_val]})
                elif isinstance(result, (list, tuple)):
                    # List/tuple to DataFrame
                    result = pd.DataFrame({'result': result})
                elif isinstance(result, dict):
                    # Dict to DataFrame
                    result = pd.DataFrame([result])
                elif hasattr(result, '__iter__') and not isinstance(result, str):
                    # Other iterable to DataFrame
                    try:
                        result = pd.DataFrame(list(result))
                    except Exception:
                        result = pd.DataFrame({'result': list(result)})
                else:
                    raise MetadataQueryError(
                        f"Query did not return a DataFrame (got {type(result).__name__})",
                        error_type='execution_error'
                    )
            
            # Limit result size
            if len(result) > self.MAX_RESULT_ROWS:
                logger.warning(f"Query returned {len(result)} rows, limiting to {self.MAX_RESULT_ROWS}")
                result = result.head(self.MAX_RESULT_ROWS)
            
            return result
            
        except SyntaxError as e:
            raise MetadataQueryError(
                f"Invalid query syntax: {str(e)}",
                error_type='syntax_error',
                original_error=e
            )
        except KeyError as e:
            raise MetadataQueryError(
                f"Column not found: {str(e)}. Available columns: {', '.join(df.columns)}",
                error_type='column_error',
                original_error=e
            )
        except TypeError as e:
            raise MetadataQueryError(
                f"Type error in query: {str(e)}",
                error_type='type_error',
                original_error=e
            )
        except Exception as e:
            raise MetadataQueryError(
                f"Query execution failed: {str(e)}",
                error_type='execution_error',
                original_error=e
            )
    
    def _generate_summary(
        self, 
        user_query: str, 
        results: pd.DataFrame, 
        query_used: str,
        data_source: str = "data"
    ) -> str:
        """
        Generate natural language summary of query results using LLM.
        
        Args:
            user_query: Original user query
            results: DataFrame with query results
            query_used: Pandas expression that was executed
            data_source: Description of the data source being queried
        
        Returns:
            Natural language summary string
        
        Raises:
            MetadataQueryError: If summary generation fails
        """
        # Build prompt for summary generation
        prompt = self._build_summary_generation_prompt(user_query, results, query_used, data_source)
        
        try:
            # Invoke LLM to generate summary
            response = self.bedrock_client.invoke_model(
                prompt=prompt,
                context=None,
                chat_history=None
            )
            
            summary = response.get('text', '').strip()
            
            if not summary:
                # Fallback summary
                summary = f"Query returned {len(results)} rows."
            
            return summary
            
        except BedrockError as e:
            raise MetadataQueryError(
                f"LLM invocation failed during summary generation: {str(e)}",
                error_type='llm_error',
                original_error=e
            )
        except Exception as e:
            raise MetadataQueryError(
                f"Summary generation failed: {str(e)}",
                error_type='summary_error',
                original_error=e
            )
    
    def _validate_query(self, query: str) -> None:
        """
        Validate pandas query for safety and correctness.
        
        Args:
            query: Query string to validate
        
        Raises:
            MetadataQueryError: If query contains unsafe operations
        """
        if not query or not query.strip():
            raise MetadataQueryError(
                "Query cannot be empty",
                error_type='validation_error'
            )
        
        # Check for unsafe operations
        query_lower = query.lower()
        for unsafe_op in self.UNSAFE_OPERATIONS:
            if unsafe_op in query_lower:
                raise MetadataQueryError(
                    f"Query contains unsafe operation: {unsafe_op}",
                    error_type='security_error'
                )
        
        # Check for common injection patterns
        dangerous_patterns = [
            r'__\w+__',  # Dunder methods
            r'import\s+',  # Import statements
            r'from\s+\w+\s+import',  # From imports
            r'os\.',  # OS module access
            r'sys\.',  # Sys module access
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise MetadataQueryError(
                    f"Query contains potentially unsafe pattern: {pattern}",
                    error_type='security_error'
                )
        
        logger.debug(f"Query validation passed: {query}")
    
    def _extract_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract schema information from DataFrame for query generation.
        
        Args:
            df: DataFrame to extract schema from
        
        Returns:
            Dictionary containing schema information
        """
        schema = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'sample_values': {},
            'null_counts': df.isnull().sum().to_dict()
        }
        
        # Get sample values for each column (first 5 unique values)
        for col in df.columns:
            try:
                unique_vals = df[col].dropna().unique()[:5].tolist()
                schema['sample_values'][col] = unique_vals
            except Exception:
                schema['sample_values'][col] = []
        
        return schema
    
    def _build_query_generation_prompt(self, user_query: str, df: pd.DataFrame) -> str:
        """
        Build prompt for LLM to generate pandas query.
        
        Args:
            user_query: User's natural language query
            df: DataFrame to query against
        
        Returns:
            Formatted prompt string
        """
        # Extract schema from the target DataFrame
        df_schema = self._extract_schema(df)
        
        prompt = f"""You are a pandas query expert. Convert the user's natural language question into a valid pandas query expression.

DataFrame Schema:
- Shape: {df_schema['shape'][0]} rows × {df_schema['shape'][1]} columns
- Columns: {', '.join(df_schema['columns'])}

Column Details:
"""
        
        for col in df_schema['columns']:
            dtype = df_schema['dtypes'].get(col, 'unknown')
            sample_vals = df_schema['sample_values'].get(col, [])
            null_count = df_schema['null_counts'].get(col, 0)
            
            prompt += f"  - {col} ({dtype})"
            if sample_vals:
                prompt += f" - Sample values: {sample_vals}"
            if null_count > 0:
                prompt += f" - {null_count} null values"
            prompt += "\n"
        
        prompt += f"""
User Question: {user_query}

Generate ONLY the pandas query expression that can be executed directly. The DataFrame is available as 'df'.

Examples of valid queries:
- df[df['column_name'] > 10]
- df.groupby('category')['value'].sum().reset_index()
- df.query("status == 'active'")
- df.sort_values('date', ascending=False).head(10)
- df[df['column'].str.contains('pattern', na=False)]

Important rules:
1. Always use 'df' as the DataFrame variable name
2. Return a DataFrame (use .reset_index() after groupby operations)
3. Use .head() to limit large result sets
4. Handle null values appropriately (use .dropna() or na=False in string operations)
5. For string comparisons, use exact matches or .str.contains()
6. Do NOT use eval(), exec(), or any system functions
7. Do NOT include explanations, only the query expression

Query:"""
        
        return prompt
    
    def _build_summary_generation_prompt(
        self, 
        user_query: str, 
        results: pd.DataFrame, 
        query_used: str,
        data_source: str = "data"
    ) -> str:
        """
        Build prompt for LLM to generate result summary.
        
        Args:
            user_query: Original user query
            results: Query results DataFrame
            query_used: Pandas expression that was executed
            data_source: Description of the data source
        
        Returns:
            Formatted prompt string
        """
        # Get preview of results (first 10 rows)
        results_preview = results.head(10).to_string()
        
        # Get basic statistics
        row_count = len(results)
        col_count = len(results.columns)
        
        prompt = f"""You are a data analyst. Summarize the following query results in natural language.

User Question: {user_query}

Data Source: {data_source}

Query Used: {query_used}

Results Summary:
- Total Rows: {row_count}
- Columns: {col_count}
- Column Names: {', '.join(results.columns.tolist())}

Results Preview (first 10 rows):
{results_preview}

Provide a concise, natural language summary that:
1. Directly answers the user's question
2. Highlights key insights, patterns, or statistics from the actual data
3. Mentions the total number of results found
4. If results are empty, explain why no data matched the criteria
5. Keep it brief and focused (2-4 sentences)

Do NOT include the query expression in your summary.

Summary:"""
        
        return prompt
    
    def _extract_query_from_response(self, response_text: str) -> str:
        """
        Extract pandas query from LLM response, removing markdown formatting.
        
        Args:
            response_text: Raw response from LLM
        
        Returns:
            Clean query string
        """
        # Remove markdown code blocks if present
        code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
        match = re.search(code_block_pattern, response_text, re.DOTALL)
        
        if match:
            query = match.group(1).strip()
        else:
            query = response_text.strip()
        
        # Remove any leading/trailing quotes
        query = query.strip('"\'')
        
        # Remove any explanatory text after the query
        # Look for common patterns like "This query..." or "Explanation:"
        lines = query.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith(('this query', 'explanation', 'note:', '#')):
                clean_lines.append(line)
        
        query = '\n'.join(clean_lines).strip()
        
        return query
