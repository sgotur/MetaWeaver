"""
S3 Metadata Loader module for loading JSON metadata from AWS S3.
Provides functionality to fetch JSON files from S3 and convert to pandas DataFrames.
"""

import json
import logging
from typing import Dict, Any, Optional, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class S3LoaderError(Exception):
    """Base exception for S3 Loader errors."""
    
    def __init__(self, message: str, error_type: str = 'general', original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class S3MetadataLoader:
    """
    Loader class for fetching JSON metadata from AWS S3.
    Handles S3 authentication, file retrieval, and DataFrame conversion.
    Supports automatic detection and flattening of nested JSON structures.
    """
    
    def __init__(self, credentials: Dict[str, str], flatten_nested: bool = True):
        """
        Initialize S3 metadata loader with AWS credentials.
        
        Args:
            credentials: Dictionary containing AWS credentials
                - aws_access_key_id: AWS access key
                - aws_secret_access_key: AWS secret key
                - region_name: AWS region
            flatten_nested: Whether to automatically flatten nested JSON structures (default: True)
        
        Raises:
            S3LoaderError: If client initialization fails
        """
        try:
            self.client = boto3.client(
                service_name='s3',
                aws_access_key_id=credentials['aws_access_key_id'],
                aws_secret_access_key=credentials['aws_secret_access_key'],
                region_name=credentials['region_name']
            )
            
            self.flatten_nested = flatten_nested
            
            logger.info(f"S3 metadata loader initialized successfully (flatten_nested={flatten_nested})")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise S3LoaderError(
                f"Failed to initialize S3 client: {str(e)}",
                error_type='initialization',
                original_error=e
            )
    
    def _detect_nested_structure(self, df: pd.DataFrame) -> bool:
        """
        Detect if DataFrame contains nested JSON structures that should be flattened.
        
        Args:
            df: pandas DataFrame to analyze
        
        Returns:
            True if nested structures are detected, False otherwise
        """
        # Check if DataFrame has only one column (common pattern for nested data)
        if len(df.columns) == 1:
            col_name = df.columns[0]
            # Check if the single column contains dict or list objects
            if df[col_name].dtype == 'object':
                sample = df[col_name].iloc[0] if len(df) > 0 else None
                if isinstance(sample, (dict, list)):
                    logger.info(f"Detected nested structure in single column '{col_name}'")
                    return True
        
        # Check for columns containing nested structures (dict or list)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Sample first non-null value
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    sample = non_null.iloc[0]
                    if isinstance(sample, (dict, list)):
                        logger.info(f"Detected nested structure in column '{col}'")
                        return True
        
        return False
    
    def _flatten_nested_dataframe(self, df: pd.DataFrame, json_data: Any) -> pd.DataFrame:
        """
        Flatten nested JSON structures in a DataFrame.
        
        Args:
            df: Original pandas DataFrame with nested structures
            json_data: Original JSON data for re-normalization
        
        Returns:
            Flattened pandas DataFrame
        
        Raises:
            S3LoaderError: If flattening fails
        """
        try:
            # Case 1: Single column with nested data (e.g., 'tables' column)
            if len(df.columns) == 1:
                col_name = df.columns[0]
                if df[col_name].dtype == 'object':
                    sample = df[col_name].iloc[0] if len(df) > 0 else None
                    
                    if isinstance(sample, dict):
                        # Column contains dictionaries - flatten them
                        logger.info(f"Flattening nested dictionaries in column '{col_name}'")
                        flattened = pd.json_normalize(df[col_name].tolist())
                        logger.info(f"Flattened to {len(flattened.columns)} columns")
                        return flattened
                    
                    elif isinstance(sample, list):
                        # Column contains lists - try to normalize
                        logger.info(f"Flattening nested lists in column '{col_name}'")
                        # Extract all list items and normalize
                        all_items = []
                        for item_list in df[col_name]:
                            if isinstance(item_list, list):
                                all_items.extend(item_list)
                        
                        if all_items and isinstance(all_items[0], dict):
                            flattened = pd.json_normalize(all_items)
                            logger.info(f"Flattened to {len(flattened.columns)} columns")
                            return flattened
            
            # Case 2: Multiple columns, some with nested structures
            # Try to normalize the original JSON data directly
            if isinstance(json_data, list):
                logger.info("Attempting to flatten nested structures in list of objects")
                flattened = pd.json_normalize(json_data)
                
                # Check if flattening actually expanded the structure
                if len(flattened.columns) > len(df.columns):
                    logger.info(f"Successfully flattened from {len(df.columns)} to {len(flattened.columns)} columns")
                    return flattened
            
            elif isinstance(json_data, dict):
                # Try to find the main data array in the dict
                for key in ['records', 'data', 'items', 'results']:
                    if key in json_data and isinstance(json_data[key], list):
                        logger.info(f"Attempting to flatten nested structures in '{key}' array")
                        flattened = pd.json_normalize(json_data[key])
                        
                        if len(flattened.columns) > len(df.columns):
                            logger.info(f"Successfully flattened from {len(df.columns)} to {len(flattened.columns)} columns")
                            return flattened
            
            # If we get here, no flattening was possible or beneficial
            logger.info("No beneficial flattening detected, returning original DataFrame")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to flatten nested structures: {str(e)}")
            raise S3LoaderError(
                f"Unable to flatten nested JSON structure. The data may be too complex or have an unsupported format. "
                f"Try setting METADATA_FLATTEN_NESTED=false to load the data as-is. Error: {str(e)}",
                error_type='flatten_failed',
                original_error=e
            )
    
    def load_metadata(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Load JSON metadata from S3 and convert to pandas DataFrame.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating if load was successful
                - dataframe: pandas DataFrame (if successful)
                - error: Error message (if failed)
                - metadata: Dictionary with row count, columns, etc.
        """
        try:
            logger.info(f"Loading metadata from S3: s3://{bucket}/{key}")
            
            # Fetch object from S3
            response = self.client.get_object(Bucket=bucket, Key=key)
            
            # Read and decode the content
            content = response['Body'].read().decode('utf-8')
            
            logger.info(f"Successfully fetched {len(content)} bytes from S3")
            
            # Parse JSON content
            try:
                json_data = json.loads(content)
                logger.info("Successfully parsed JSON content")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {str(e)}")
                raise S3LoaderError(
                    f"Invalid JSON format in S3 file: {str(e)}",
                    error_type='json_parse',
                    original_error=e
                )
            
            # Convert to DataFrame
            try:
                # Handle different JSON structures
                if isinstance(json_data, list):
                    # JSON array of objects
                    df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # Check if it's a dict with records
                    if 'records' in json_data:
                        df = pd.DataFrame(json_data['records'])
                    elif 'data' in json_data:
                        df = pd.DataFrame(json_data['data'])
                    else:
                        # Try to convert dict directly
                        df = pd.DataFrame([json_data])
                else:
                    raise S3LoaderError(
                        f"Unsupported JSON structure: expected list or dict, got {type(json_data)}",
                        error_type='json_structure'
                    )
                
                logger.info(f"Successfully converted to DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Check for nested structures and flatten if enabled
                if self.flatten_nested and self._detect_nested_structure(df):
                    logger.info("Nested structure detected, attempting to flatten...")
                    try:
                        original_shape = df.shape
                        df = self._flatten_nested_dataframe(df, json_data)
                        
                        if df.shape != original_shape:
                            logger.info(
                                f"Successfully flattened nested structure: "
                                f"{original_shape[0]} rows × {original_shape[1]} cols → "
                                f"{df.shape[0]} rows × {df.shape[1]} cols"
                            )
                    except S3LoaderError as e:
                        # If flattening fails, log warning but continue with original DataFrame
                        logger.warning(f"Flattening failed, using original structure: {str(e)}")
                        # Re-raise if it's a critical error
                        if e.error_type == 'flatten_failed':
                            raise
                
                # Extract metadata
                metadata = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    's3_uri': f"s3://{bucket}/{key}",
                    'file_size_bytes': len(content),
                    'flattened': self.flatten_nested and self._detect_nested_structure(df) is False  # Was flattened
                }
                
                return {
                    'success': True,
                    'dataframe': df,
                    'error': None,
                    'metadata': metadata
                }
                
            except Exception as e:
                logger.error(f"Failed to convert JSON to DataFrame: {str(e)}")
                raise S3LoaderError(
                    f"Failed to convert JSON to DataFrame: {str(e)}",
                    error_type='dataframe_conversion',
                    original_error=e
                )
        
        except NoCredentialsError as e:
            logger.error("AWS credentials not found or invalid")
            return {
                'success': False,
                'dataframe': None,
                'error': "AWS credentials not found or invalid. Please check your configuration.",
                'metadata': {}
            }
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            # Handle specific S3 errors
            if error_code == 'NoSuchBucket':
                logger.error(f"S3 bucket not found: {bucket}")
                return {
                    'success': False,
                    'dataframe': None,
                    'error': f"S3 bucket '{bucket}' not found. Please verify the bucket name in configuration.",
                    'metadata': {}
                }
            elif error_code == 'NoSuchKey':
                logger.error(f"S3 object not found: {key}")
                return {
                    'success': False,
                    'dataframe': None,
                    'error': f"File '{key}' not found in bucket '{bucket}'. Please verify the file path.",
                    'metadata': {}
                }
            elif error_code in ['AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
                logger.error(f"S3 access denied: {error_message}")
                return {
                    'success': False,
                    'dataframe': None,
                    'error': f"Access denied to S3. Please check your AWS credentials and permissions.",
                    'metadata': {}
                }
            else:
                logger.error(f"S3 error: {error_code} - {error_message}")
                return {
                    'success': False,
                    'dataframe': None,
                    'error': f"S3 error: {error_message}",
                    'metadata': {}
                }
        
        except S3LoaderError as e:
            # Re-raise our custom errors with proper structure
            logger.error(f"S3 loader error: {str(e)}")
            return {
                'success': False,
                'dataframe': None,
                'error': str(e),
                'metadata': {}
            }
        
        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected error loading metadata: {str(e)}")
            return {
                'success': False,
                'dataframe': None,
                'error': f"Unexpected error: {str(e)}",
                'metadata': {}
            }
    
    def get_dataframe_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract schema information from a pandas DataFrame.
        
        Args:
            df: pandas DataFrame to analyze
        
        Returns:
            Dictionary containing:
                - columns: List of column names
                - dtypes: Dictionary mapping column names to data types
                - shape: Tuple of (rows, columns)
                - sample_values: Dictionary with first 5 values per column
                - null_counts: Dictionary with null count per column
        """
        try:
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'shape': df.shape,
                'null_counts': df.isnull().sum().to_dict()
            }
            
            # Extract sample values (first 5 non-null values per column)
            sample_values = {}
            for col in df.columns:
                # Get first 5 non-null values
                non_null_values = df[col].dropna().head(5).tolist()
                sample_values[col] = non_null_values
            
            schema['sample_values'] = sample_values
            
            logger.info(f"Extracted schema for DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
            
            return schema
            
        except Exception as e:
            logger.error(f"Failed to extract DataFrame schema: {str(e)}")
            raise S3LoaderError(
                f"Failed to extract DataFrame schema: {str(e)}",
                error_type='schema_extraction',
                original_error=e
            )
