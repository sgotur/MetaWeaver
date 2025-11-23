"""
Configuration management module for Bedrock Chatbot Application.
Loads and validates environment variables from .env file.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class Settings:
    """Configuration settings manager for the application."""
    
    def __init__(self):
        """Initialize settings by loading environment variables."""
        self._load_environment()
        self._validate_configuration()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        # Load .env file if it exists
        if not load_dotenv():
            # Check if .env file exists
            if not os.path.exists('.env'):
                raise ConfigurationError(
                    "Configuration file '.env' not found. "
                    "Please create a .env file based on .env.example template."
                )
    
    def _validate_configuration(self) -> None:
        """Validate that all required environment variables are present."""
        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'AWS_REGION',
            'BEDROCK_MODEL_ID',
            'KNOWLEDGE_BASE_ID'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.strip() == '':
                missing_vars.append(var)
        
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please ensure all required variables are set in your .env file."
            )
    
    def _validate_metadata_configuration(self) -> None:
        """Validate metadata-specific configuration variables."""
        required_metadata_vars = [
            'METADATA_S3_BUCKET',
            'METADATA_S3_KEY'
        ]
        
        missing_vars = []
        for var in required_metadata_vars:
            value = os.getenv(var)
            if not value or value.strip() == '':
                missing_vars.append(var)
        
        if missing_vars:
            raise ConfigurationError(
                f"Missing required metadata environment variables: {', '.join(missing_vars)}. "
                f"Please ensure all required variables are set in your .env file."
            )
    
    def get_aws_credentials(self) -> Dict[str, str]:
        """
        Get AWS credentials from environment variables.
        
        Returns:
            Dictionary containing AWS credentials
            
        Raises:
            ConfigurationError: If credentials are missing or invalid
        """
        credentials = {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region_name': os.getenv('AWS_REGION')
        }
        
        # Validate credentials are not empty
        for key, value in credentials.items():
            if not value or value.strip() == '':
                raise ConfigurationError(
                    f"AWS credential '{key}' is empty or invalid. "
                    f"Please check your .env file."
                )
        
        return credentials
    
    def get_bedrock_config(self) -> Dict[str, Any]:
        """
        Get Bedrock-specific configuration.
        
        Returns:
            Dictionary containing Bedrock configuration parameters
            
        Raises:
            ConfigurationError: If configuration is missing or invalid
        """
        model_id = os.getenv('BEDROCK_MODEL_ID')
        kb_id = os.getenv('KNOWLEDGE_BASE_ID')
        
        if not model_id or model_id.strip() == '':
            raise ConfigurationError(
                "BEDROCK_MODEL_ID is not set. Please check your .env file."
            )
        
        if not kb_id or kb_id.strip() == '':
            raise ConfigurationError(
                "KNOWLEDGE_BASE_ID is not set. Please check your .env file."
            )
        
        # Get optional parameters with defaults
        try:
            max_tokens = int(os.getenv('MAX_TOKENS', '2048'))
        except ValueError:
            raise ConfigurationError(
                "MAX_TOKENS must be a valid integer."
            )
        
        try:
            temperature = float(os.getenv('TEMPERATURE', '0.7'))
            if not 0 <= temperature <= 1:
                raise ValueError()
        except ValueError:
            raise ConfigurationError(
                "TEMPERATURE must be a valid float between 0 and 1."
            )
        
        return {
            'model_id': model_id,
            'knowledge_base_id': kb_id,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
    
    def get_metadata_config(self) -> Dict[str, Any]:
        """
        Get metadata-specific configuration.
        
        Returns:
            Dictionary containing metadata configuration parameters
            
        Raises:
            ConfigurationError: If configuration is missing or invalid
        """
        # Validate metadata configuration
        self._validate_metadata_configuration()
        
        s3_bucket = os.getenv('METADATA_S3_BUCKET')
        s3_key = os.getenv('METADATA_S3_KEY')
        
        if not s3_bucket or s3_bucket.strip() == '':
            raise ConfigurationError(
                "METADATA_S3_BUCKET is not set. Please check your .env file."
            )
        
        if not s3_key or s3_key.strip() == '':
            raise ConfigurationError(
                "METADATA_S3_KEY is not set. Please check your .env file."
            )
        
        # Get optional parameters with defaults
        try:
            max_result_rows = int(os.getenv('MAX_RESULT_ROWS', '10000'))
            if max_result_rows <= 0:
                raise ValueError()
        except ValueError:
            raise ConfigurationError(
                "MAX_RESULT_ROWS must be a valid positive integer."
            )
        
        try:
            query_timeout_seconds = int(os.getenv('QUERY_TIMEOUT_SECONDS', '30'))
            if query_timeout_seconds <= 0:
                raise ValueError()
        except ValueError:
            raise ConfigurationError(
                "QUERY_TIMEOUT_SECONDS must be a valid positive integer."
            )
        
        # Get flatten_nested option (default: True)
        flatten_nested_str = os.getenv('METADATA_FLATTEN_NESTED', 'true').lower()
        flatten_nested = flatten_nested_str in ['true', '1', 'yes', 'on']
        
        # Get CSV data configuration (optional, defaults to metadata bucket)
        csv_data_bucket = os.getenv('CSV_DATA_S3_BUCKET', s3_bucket)
        csv_data_prefix = os.getenv('CSV_DATA_S3_PREFIX', 'Data/')
        
        return {
            's3_bucket': s3_bucket,
            's3_key': s3_key,
            'max_result_rows': max_result_rows,
            'query_timeout_seconds': query_timeout_seconds,
            'flatten_nested': flatten_nested,
            'csv_data_bucket': csv_data_bucket,
            'csv_data_prefix': csv_data_prefix
        }
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get complete configuration including AWS credentials and Bedrock settings.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'aws': self.get_aws_credentials(),
            'bedrock': self.get_bedrock_config()
        }


def load_config() -> Settings:
    """
    Load and validate application configuration.
    
    Returns:
        Settings object with validated configuration
        
    Raises:
        ConfigurationError: If configuration is missing or invalid
    """
    try:
        settings = Settings()
        return settings
    except ConfigurationError as e:
        # Re-raise with additional context
        raise ConfigurationError(
            f"Failed to load configuration: {str(e)}"
        ) from e
    except Exception as e:
        # Catch any unexpected errors
        raise ConfigurationError(
            f"Unexpected error loading configuration: {str(e)}"
        ) from e
