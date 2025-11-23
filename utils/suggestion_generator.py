"""
Question suggestion generator module for Bedrock Chatbot Application.
Generates contextually relevant follow-up questions based on conversation history.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from utils.aws_client import BedrockClient, BedrockError

# Configure logging
logger = logging.getLogger(__name__)


class SuggestionGenerator:
    """
    Generates contextually relevant follow-up questions based on conversation history.
    Uses Claude Sonnet 4.5 to analyze recent conversation turns and suggest relevant questions.
    """
    
    # Number of recent conversation turns to analyze
    CONTEXT_TURNS = 3
    
    # Default number of suggestions to generate
    DEFAULT_NUM_SUGGESTIONS = 4
    
    # Maximum character length for suggestions
    MAX_SUGGESTION_LENGTH = 100
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize SuggestionGenerator with BedrockClient.
        
        Args:
            bedrock_client: BedrockClient instance for model invocation
        """
        self.bedrock_client = bedrock_client
        logger.info("SuggestionGenerator initialized successfully")
    
    def generate_suggestions(
        self, 
        conversation_context: List[Dict[str, str]], 
        num_suggestions: int = DEFAULT_NUM_SUGGESTIONS
    ) -> List[str]:
        """
        Generate follow-up questions based on conversation context.
        
        This method:
        1. Extracts last 2-3 conversation turns for context
        2. Builds a prompt for suggestion generation
        3. Invokes Claude Sonnet 4.5 to generate questions
        4. Parses and validates the suggestions
        5. Returns list of 3-5 suggestion strings
        
        Args:
            conversation_context: List of message dictionaries with 'role' and 'content' keys
            num_suggestions: Number of suggestions to generate (default: 4)
        
        Returns:
            List of suggestion strings (questions)
            Returns empty list if generation fails
        """
        try:
            logger.info(f"Generating {num_suggestions} suggestions from conversation context")
            
            # Step 1: Extract recent conversation turns
            recent_context = self._extract_recent_context(conversation_context)
            
            if not recent_context:
                logger.warning("No conversation context available for suggestion generation")
                return []
            
            # Step 2: Build suggestion prompt
            suggestion_prompt = self._create_suggestion_prompt(recent_context, num_suggestions)
            
            # Step 3: Invoke model to generate suggestions
            try:
                model_response = self.bedrock_client.invoke_model(
                    prompt=suggestion_prompt,
                    context=None,
                    chat_history=None
                )
                
                response_text = model_response.get('text', '')
                logger.debug(f"Model response for suggestions: {response_text[:200]}...")
                
            except BedrockError as e:
                logger.error(f"Bedrock error during suggestion generation: {str(e)}")
                return []
            
            # Step 4: Parse suggestions from response
            suggestions = self._parse_suggestions(response_text)
            
            # Step 5: Validate and format suggestions
            validated_suggestions = self._validate_and_format_suggestions(suggestions, num_suggestions)
            
            logger.info(f"Successfully generated {len(validated_suggestions)} suggestions")
            return validated_suggestions
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_suggestions: {str(e)}")
            return []
    
    def _extract_recent_context(self, conversation_context: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Extract the last 2-3 conversation turns for suggestion generation.
        
        Args:
            conversation_context: Full conversation history
        
        Returns:
            List of recent message dictionaries
        """
        if not conversation_context:
            return []
        
        # Calculate how many messages to include (last CONTEXT_TURNS exchanges)
        # Each turn is typically 2 messages (user + assistant)
        num_messages = self.CONTEXT_TURNS * 2
        
        recent_messages = conversation_context[-num_messages:] if len(conversation_context) > num_messages else conversation_context
        
        logger.debug(f"Extracted {len(recent_messages)} messages from conversation context")
        return recent_messages
    
    def _create_suggestion_prompt(self, context: List[Dict[str, str]], num_suggestions: int) -> str:
        """
        Build a prompt for the model to generate follow-up questions.
        
        Args:
            context: Recent conversation messages
            num_suggestions: Number of suggestions to generate
        
        Returns:
            Formatted prompt string
        """
        # Build conversation summary
        conversation_summary = []
        for msg in context:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            if role == 'user':
                conversation_summary.append(f"User: {content}")
            elif role == 'assistant':
                conversation_summary.append(f"Assistant: {content}")
        
        conversation_text = "\n".join(conversation_summary)
        
        # Create the suggestion prompt
        prompt = f"""Based on the following conversation, generate {num_suggestions} relevant follow-up questions that the user might want to ask next.

Conversation:
{conversation_text}

Generate {num_suggestions} concise, specific follow-up questions that:
1. Are directly related to the conversation topic
2. Help the user explore the topic more deeply
3. Are clear and easy to understand
4. Are under 100 characters each
5. Are diverse and cover different aspects

Format your response as a numbered list with one question per line:
1. [First question]
2. [Second question]
3. [Third question]
etc.

Only provide the questions, no additional explanation."""
        
        return prompt
    
    def _parse_suggestions(self, response_text: str) -> List[str]:
        """
        Parse individual questions from the model response.
        
        Args:
            response_text: Raw text response from the model
        
        Returns:
            List of extracted question strings
        """
        suggestions = []
        
        # Try to extract numbered list items
        # Pattern matches: "1. Question text" or "1) Question text"
        pattern = r'^\s*\d+[\.\)]\s*(.+?)$'
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                question = match.group(1).strip()
                if question:
                    suggestions.append(question)
            elif line and not any(line.startswith(prefix) for prefix in ['Based on', 'Here are', 'Follow-up', 'Questions:']):
                # If line doesn't match pattern but looks like a question, include it
                if '?' in line or line[0].isupper():
                    suggestions.append(line)
        
        logger.debug(f"Parsed {len(suggestions)} suggestions from response")
        return suggestions
    
    def _validate_and_format_suggestions(self, suggestions: List[str], target_count: int) -> List[str]:
        """
        Validate and format suggestions according to requirements.
        
        This method:
        1. Validates that suggestions are properly formatted questions
        2. Removes duplicate or overly similar suggestions
        3. Ensures suggestions are concise (under 100 characters)
        4. Returns the target number of suggestions
        
        Args:
            suggestions: Raw list of suggestion strings
            target_count: Desired number of suggestions
        
        Returns:
            List of validated and formatted suggestions
        """
        validated = []
        seen_lowercase = set()
        
        for suggestion in suggestions:
            # Clean up the suggestion
            cleaned = suggestion.strip()
            
            # Remove any leading/trailing quotes
            cleaned = cleaned.strip('"\'')
            
            # Ensure it ends with a question mark if it looks like a question
            if cleaned and not cleaned.endswith('?') and any(word in cleaned.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']):
                cleaned += '?'
            
            # Validate: must be a proper question
            if not self._is_valid_question(cleaned):
                logger.debug(f"Skipping invalid question: {cleaned}")
                continue
            
            # Validate: must be under MAX_SUGGESTION_LENGTH characters
            if len(cleaned) > self.MAX_SUGGESTION_LENGTH:
                logger.debug(f"Skipping too long suggestion ({len(cleaned)} chars): {cleaned[:50]}...")
                continue
            
            # Check for duplicates (case-insensitive)
            cleaned_lower = cleaned.lower()
            if cleaned_lower in seen_lowercase:
                logger.debug(f"Skipping duplicate suggestion: {cleaned}")
                continue
            
            # Check for overly similar suggestions (simple similarity check)
            if self._is_too_similar(cleaned_lower, seen_lowercase):
                logger.debug(f"Skipping similar suggestion: {cleaned}")
                continue
            
            # Add to validated list
            validated.append(cleaned)
            seen_lowercase.add(cleaned_lower)
            
            # Stop if we have enough suggestions
            if len(validated) >= target_count:
                break
        
        logger.info(f"Validated {len(validated)} out of {len(suggestions)} suggestions")
        return validated
    
    def _is_valid_question(self, text: str) -> bool:
        """
        Check if text is a properly formatted question.
        
        Args:
            text: Text to validate
        
        Returns:
            True if text is a valid question, False otherwise
        """
        if not text or len(text) < 10:
            return False
        
        # Must end with question mark
        if not text.endswith('?'):
            return False
        
        # Must start with capital letter or question word
        if not text[0].isupper():
            return False
        
        # Should contain at least one question word or be interrogative
        question_indicators = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'will', 'did']
        text_lower = text.lower()
        
        has_question_word = any(text_lower.startswith(word) or f' {word} ' in text_lower for word in question_indicators)
        
        return has_question_word
    
    def _is_too_similar(self, text: str, existing_texts: set) -> bool:
        """
        Check if text is too similar to existing suggestions.
        Uses simple word overlap heuristic.
        
        Args:
            text: Text to check
            existing_texts: Set of existing suggestion texts (lowercase)
        
        Returns:
            True if text is too similar to any existing text, False otherwise
        """
        # Extract words from the new text
        words = set(text.split())
        
        # Remove common question words for comparison
        common_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'about', '?'}
        words = words - common_words
        
        if not words:
            return False
        
        # Check similarity with each existing text
        for existing in existing_texts:
            existing_words = set(existing.split()) - common_words
            
            if not existing_words:
                continue
            
            # Calculate word overlap ratio
            overlap = len(words & existing_words)
            min_length = min(len(words), len(existing_words))
            
            if min_length > 0:
                similarity = overlap / min_length
                
                # If more than 60% of words overlap, consider it too similar
                if similarity > 0.6:
                    return True
        
        return False
