"""
CSV Data Loader module for loading CSV files from AWS S3.
Provides functionality to fetch CSV files from S3 and convert to pandas DataFrames.
"""

import logging
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import io

# Configure logging
logger = logging.getLogger(__name__)


class CSVLoaderError(Exception):
    """Base exception for CSV Loader errors."""
    
    def __init__(self, message: str, error_type: str = 'general', original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class CSVDataLoader:
    """
    Loader class for fetching CSV data files from AWS S3.
    Handles S3 authentication, file retrieval, and DataFrame conversion.
    """
    
    def __init__(self, credentials: Dict[str, str], bucket: str, prefix: str = ''):
        """
        Initialize CSV data loader with AWS credentials and S3 location.
        
        Args:
            credentials: Dictionary containing AWS credentials
                - aws_access_key_id: AWS access key
                - aws_secret_access_key: AWS secret key
                - region_name: AWS region
            bucket: S3 bucket name where CSV files are stored
            prefix: Optional S3 prefix/folder path (e.g., 'Data/')
        
        Raises:
            CSVLoaderError: If client initialization fails
        """
        try:
            self.client = boto3.client(
                service_name='s3',
                aws_access_key_id=credentials['aws_access_key_id'],
                aws_secret_access_key=credentials['aws_secret_access_key'],
                region_name=credentials['region_name']
            )
            
            self.bucket = bucket
            self.prefix = prefix.rstrip('/') + '/' if prefix and not prefix.endswith('/') else prefix
            
            logger.info(f"CSV data loader initialized successfully (bucket: {bucket}, prefix: {prefix})")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise CSVLoaderError(
                f"Failed to initialize S3 client: {str(e)}",
                error_type='initialization',
                original_error=e
            )

    
    def load_csv_data(self, filename: str) -> Dict[str, Any]:
        """
        Load CSV data from S3 and convert to pandas DataFrame.
        
        Args:
            filename: CSV filename (e.g., 't_employee.csv')
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating if load was successful
                - dataframe: pandas DataFrame (if successful)
                - error: Error message (if failed)
                - metadata: Dictionary with row count, columns, file info, etc.
        """
        # Construct full S3 key
        key = f"{self.prefix}{filename}" if self.prefix else filename
        
        try:
            logger.info(f"Loading CSV data from S3: s3://{self.bucket}/{key}")
            
            # Fetch object from S3
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            
            # Read and decode the content
            content = response['Body'].read()
            file_size = len(content)
            
            logger.info(f"Successfully fetched {file_size} bytes from S3")
            
            # Parse CSV content
            try:
                # Use io.BytesIO to read CSV from bytes
                df = pd.read_csv(io.BytesIO(content))
                logger.info(f"Successfully parsed CSV: {df.shape[0]} rows, {df.shape[1]} columns")
                
            except Exception as e:
                logger.error(f"Failed to parse CSV: {str(e)}")
                raise CSVLoaderError(
                    f"Invalid CSV format in S3 file: {str(e)}",
                    error_type='csv_parse',
                    original_error=e
                )
            
            # Extract metadata
            metadata = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                's3_uri': f"s3://{self.bucket}/{key}",
                'file_size_bytes': file_size,
                'filename': filename
            }
            
            return {
                'success': True,
                'dataframe': df,
                'error': None,
                'metadata': metadata
            }
        
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
                logger.error(f"S3 bucket not found: {self.bucket}")
                return {
                    'success': False,
                    'dataframe': None,
                    'error': f"S3 bucket '{self.bucket}' not found. Please verify the bucket name in configuration.",
                    'metadata': {}
                }
            elif error_code == 'NoSuchKey':
                logger.error(f"S3 object not found: {key}")
                return {
                    'success': False,
                    'dataframe': None,
                    'error': f"CSV file '{filename}' not found in bucket '{self.bucket}' at path '{self.prefix}'. Please verify the file exists.",
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
        
        except CSVLoaderError as e:
            # Re-raise our custom errors with proper structure
            logger.error(f"CSV loader error: {str(e)}")
            return {
                'success': False,
                'dataframe': None,
                'error': str(e),
                'metadata': {}
            }
        
        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected error loading CSV: {str(e)}")
            return {
                'success': False,
                'dataframe': None,
                'error': f"Unexpected error: {str(e)}",
                'metadata': {}
            }
    
    def get_dataframe_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract detailed information from a pandas DataFrame.
        
        Args:
            df: pandas DataFrame to analyze
        
        Returns:
            Dictionary containing:
                - columns: List of column names
                - dtypes: Dictionary mapping column names to data types
                - shape: Tuple of (rows, columns)
                - sample_values: Dictionary with first 5 values per column
                - null_counts: Dictionary with null count per column
                - memory_usage: Memory usage in bytes
        """
        try:
            info = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'shape': df.shape,
                'null_counts': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Extract sample values (first 5 non-null values per column)
            sample_values = {}
            for col in df.columns:
                # Get first 5 non-null values
                non_null_values = df[col].dropna().head(5).tolist()
                sample_values[col] = non_null_values
            
            info['sample_values'] = sample_values
            
            logger.info(f"Extracted info for DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to extract DataFrame info: {str(e)}")
            raise CSVLoaderError(
                f"Failed to extract DataFrame info: {str(e)}",
                error_type='info_extraction',
                original_error=e
            )
