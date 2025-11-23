"""
Metadata Query Chat interface page for Bedrock Chatbot Application.
Provides interactive chat UI for querying structured metadata using natural language.
"""

import os
import warnings
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Configure logging to suppress ScriptRunContext warnings  
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')

import streamlit as st
from config.settings import load_config, ConfigurationError
from utils.aws_client import BedrockClient, BedrockError
from utils.s3_loader import S3MetadataLoader, S3LoaderError
from utils.csv_data_loader import CSVDataLoader, CSVLoaderError
from utils.metadata_query_handler import MetadataQueryHandler, MetadataQueryError
from utils.suggestion_generator import SuggestionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Metadata Query Chat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS from external file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar navigation menu with icons
st.sidebar.title("MetaBeaver")
st.sidebar.markdown("---")

# Navigation menu items
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.switch_page("Home.py")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("Metadata Query Chat v1.0\n\nQuery structured data with natural language")


def display_error(error_type: str, error_message: str, show_retry: bool = True):
    """
    Display contextual error messages in Streamlit with user-friendly explanations.
    Logs errors for troubleshooting while showing clean UI messages.
    
    Args:
        error_type: Type of error (auth, service, network, model, s3, query, general)
        error_message: Detailed error message
        show_retry: Whether to show retry suggestion (default: True)
    """
    # Log the full error for troubleshooting
    logger.error(f"Error displayed to user - Type: {error_type}, Message: {error_message}")
    
    if error_type == 'auth':
        st.error("🔐 **Authentication Error**")
        st.markdown("""
        There was a problem authenticating with AWS. This usually means:
        - Your AWS credentials are invalid or expired
        - Your credentials don't have the necessary permissions
        
        **What to do:**
        1. Check your `.env` file and verify your AWS credentials
        2. Ensure your AWS access key and secret key are correct
        3. Verify your IAM user/role has permissions for:
           - `bedrock:InvokeModel`
           - `s3:GetObject` (for metadata access)
        4. If using temporary credentials, check if they've expired
        """)
        
        with st.expander("🔍 Technical Details"):
            st.code(error_message, language=None)
    
    elif error_type == 's3':
        st.error("📦 **S3 Access Error**")
        st.markdown("""
        There was a problem accessing the metadata file in S3. This could be due to:
        - Incorrect S3 bucket or key configuration
        - Missing S3 permissions
        - File not found in the specified location
        
        **What to do:**
        1. Verify `METADATA_S3_BUCKET` and `METADATA_S3_KEY` in your `.env` file
        2. Ensure the file exists at the specified S3 location
        3. Check that your AWS credentials have `s3:GetObject` permission
        4. Verify the bucket and key are correct
        """)
        
        with st.expander("🔍 Technical Details"):
            st.code(error_message, language=None)
    
    elif error_type == 'query':
        st.warning("⚠️ **Query Error**")
        st.markdown("""
        There was a problem processing your query. This could be due to:
        - Invalid query syntax
        - Column names that don't exist in the data
        - Data type mismatches
        
        **What to do:**
        1. Try rephrasing your question
        2. Check the available columns in the sidebar
        3. Make sure you're referencing valid column names
        4. Try a simpler query first
        """)
        
        if show_retry:
            st.info("💡 **Tip:** Try asking about the data structure first, like 'What columns are available?'")
        
        with st.expander("🔍 Technical Details"):
            st.code(error_message, language=None)
    
    elif error_type == 'service':
        st.warning("⚠️ **Service Temporarily Unavailable**")
        st.markdown("""
        The AWS Bedrock service is currently unavailable. This could be due to:
        - Temporary service outage
        - Rate limiting (too many requests)
        - Regional service issues
        
        **What to do:**
        1. Wait a moment and try your message again
        2. Check the [AWS Service Health Dashboard](https://status.aws.amazon.com/)
        3. If the issue persists, try again in a few minutes
        """)
        
        if show_retry:
            st.info("💡 **Tip:** The system automatically retries failed requests, but you can try sending your message again.")
        
        with st.expander("🔍 Technical Details"):
            st.code(error_message, language=None)
    
    elif error_type == 'network':
        st.info("🌐 **Connection Issue**")
        st.markdown("""
        There was a problem connecting to AWS services. This could be due to:
        - Network connectivity issues
        - Firewall or proxy blocking the connection
        - DNS resolution problems
        - Request timeout
        
        **What to do:**
        1. Check your internet connection
        2. Verify you can access AWS services from your network
        3. If behind a corporate firewall, check proxy settings
        4. Try your message again
        """)
        
        if show_retry:
            st.info("💡 **Tip:** Click the retry button below or simply send your message again.")
        
        with st.expander("🔍 Technical Details"):
            st.code(error_message, language=None)
    
    elif error_type == 'model':
        st.error("🤖 **Model Error**")
        st.markdown("""
        The AI model encountered an issue processing your request. This could be due to:
        - Message is too long (exceeds token limit)
        - Invalid or malformed input
        - Model-specific processing error
        
        **What to do:**
        1. Try making your message shorter or simpler
        2. Rephrase your question
        3. Break complex questions into smaller parts
        4. If the issue persists, try starting a new conversation
        """)
        
        with st.expander("🔍 Technical Details"):
            st.code(error_message, language=None)
    
    else:
        st.error("❌ **An Error Occurred**")
        st.markdown("""
        An unexpected error occurred while processing your request.
        
        **What to do:**
        1. Try sending your message again
        2. If the issue persists, try refreshing the page
        3. Check the technical details below for more information
        """)
        
        with st.expander("🔍 Technical Details"):
            st.code(error_message, language=None)


def initialize_session_state():
    """Initialize session state variables for the metadata query chat interface."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'metadata_df' not in st.session_state:
        st.session_state.metadata_df = None
    
    if 'metadata_loaded' not in st.session_state:
        st.session_state.metadata_loaded = False
    
    if 'metadata_error' not in st.session_state:
        st.session_state.metadata_error = None
    
    if 'metadata_info' not in st.session_state:
        st.session_state.metadata_info = {}
    
    if 'query_handler' not in st.session_state:
        st.session_state.query_handler = None
    
    if 'config_error' not in st.session_state:
        st.session_state.config_error = None
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if 'suggestion_generator' not in st.session_state:
        st.session_state.suggestion_generator = None
    
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []


@st.cache_resource
def get_cached_configuration():
    """
    Load and cache configuration to avoid repeated file reads.
    Uses Streamlit's cache_resource to persist across reruns.
    
    Returns:
        Tuple of (settings, error_message, error_type)
    """
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            return None, "Configuration file '.env' not found.", 'config'
        
        # Try to load and validate configuration
        settings = load_config()
        
        # Validate AWS credentials are present
        aws_creds = settings.get_aws_credentials()
        
        # Validate Bedrock configuration
        bedrock_config = settings.get_bedrock_config()
        
        # Validate metadata configuration
        metadata_config = settings.get_metadata_config()
        
        return settings, None, None
        
    except ConfigurationError as e:
        return None, str(e), 'config'
    except Exception as e:
        return None, f"Unexpected configuration error: {str(e)}", 'config'


def validate_startup_configuration():
    """
    Validate configuration on page startup before initializing components.
    Returns tuple of (is_valid, error_message, error_type)
    """
    settings, error_message, error_type = get_cached_configuration()
    
    if settings is None:
        return False, error_message, error_type
    
    return True, None, None


def load_metadata_from_s3():
    """
    Load metadata from S3 and cache in session state.
    Returns True if successful, False otherwise.
    """
    if st.session_state.metadata_loaded and st.session_state.metadata_df is not None:
        return True
    
    try:
        # Get configuration
        settings, _, _ = get_cached_configuration()
        if settings is None:
            st.session_state.metadata_error = "Configuration not available"
            return False
        
        aws_credentials = settings.get_aws_credentials()
        metadata_config = settings.get_metadata_config()
        
        # Initialize S3 loader with flatten_nested option
        s3_loader = S3MetadataLoader(
            credentials=aws_credentials,
            flatten_nested=metadata_config.get('flatten_nested', True)
        )
        
        # Load metadata
        logger.info(f"Loading metadata from S3: {metadata_config['s3_bucket']}/{metadata_config['s3_key']}")
        result = s3_loader.load_metadata(
            bucket=metadata_config['s3_bucket'],
            key=metadata_config['s3_key']
        )
        
        if result['success']:
            st.session_state.metadata_df = result['dataframe']
            st.session_state.metadata_info = result['metadata']
            st.session_state.metadata_loaded = True
            st.session_state.metadata_error = None
            logger.info(f"Metadata loaded successfully: {result['metadata']['row_count']} rows")
            return True
        else:
            st.session_state.metadata_error = result['error']
            st.session_state.metadata_loaded = False
            logger.error(f"Failed to load metadata: {result['error']}")
            return False
            
    except S3LoaderError as e:
        error_msg = f"S3 loader error: {str(e)}"
        st.session_state.metadata_error = error_msg
        st.session_state.metadata_loaded = False
        logger.error(error_msg)
        return False
    except Exception as e:
        error_msg = f"Unexpected error loading metadata: {str(e)}"
        st.session_state.metadata_error = error_msg
        st.session_state.metadata_loaded = False
        logger.error(error_msg)
        return False


@st.cache_resource
def get_query_handler(_metadata_df):
    """
    Initialize and cache MetadataQueryHandler with CSV data loader.
    Uses Streamlit's cache_resource to persist across reruns.
    
    Args:
        _metadata_df: pandas DataFrame (underscore prefix to exclude from hashing)
    
    Returns:
        Tuple of (query_handler, error_message)
    """
    try:
        # Load configuration from cache
        settings, error_message, error_type = get_cached_configuration()
        
        if settings is None:
            return None, error_message
        
        aws_credentials = settings.get_aws_credentials()
        bedrock_config = settings.get_bedrock_config()
        metadata_config = settings.get_metadata_config()
        
        # Initialize Bedrock client
        bedrock_client = BedrockClient(
            credentials=aws_credentials,
            model_id=bedrock_config['model_id'],
            max_tokens=bedrock_config['max_tokens'],
            temperature=bedrock_config['temperature']
        )
        
        # Initialize CSV data loader
        csv_loader = CSVDataLoader(
            credentials=aws_credentials,
            bucket=metadata_config.get('csv_data_bucket'),
            prefix=metadata_config.get('csv_data_prefix', '')
        )
        
        # Initialize MetadataQueryHandler with CSV loader
        query_handler = MetadataQueryHandler(
            bedrock_client=bedrock_client,
            metadata_df=_metadata_df,
            csv_loader=csv_loader
        )
        
        logger.info("MetadataQueryHandler initialized with CSV data loader and cached successfully")
        return query_handler, None
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        return None, str(e)
    except MetadataQueryError as e:
        logger.error(f"Query handler initialization error: {str(e)}")
        return None, str(e)
    except CSVLoaderError as e:
        logger.error(f"CSV loader initialization error: {str(e)}")
        return None, str(e)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {str(e)}")
        return None, f"Unexpected initialization error: {str(e)}"


def initialize_query_handler():
    """
    Initialize MetadataQueryHandler after metadata is loaded.
    Returns True if successful, False otherwise.
    """
    if st.session_state.query_handler is not None:
        return True
    
    if st.session_state.metadata_df is None:
        return False
    
    # Get cached query handler
    query_handler, error_message = get_query_handler(st.session_state.metadata_df)
    
    if query_handler is None:
        st.session_state.config_error = error_message
        return False
    
    st.session_state.query_handler = query_handler
    st.session_state.initialized = True
    
    # Initialize suggestion generator with the same bedrock client
    if st.session_state.suggestion_generator is None:
        try:
            bedrock_client = query_handler.bedrock_client
            st.session_state.suggestion_generator = SuggestionGenerator(bedrock_client)
            logger.info("SuggestionGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SuggestionGenerator: {str(e)}")
    
    return True


def display_chat_messages():
    """Display all messages in the chat history with enhanced formatting."""
    for message in st.session_state.messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        with st.chat_message(role):
            # Display message content with markdown support
            st.markdown(content, unsafe_allow_html=False)
            
            # For assistant messages, display query results if available
            if role == 'assistant':
                data_source = message.get('data_source')
                query_results = message.get('query_results')
                query_used = message.get('query_used')
                
                # Display data source info (commented out)
                # if data_source and data_source != "metadata catalog":
                #     st.info(f"📊 **Data Source:** {data_source}")
                
                if query_results is not None and not query_results.empty:
                    display_query_results(query_results, query_used)
            
            # Display timestamp if available
            timestamp = message.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%I:%M %p")
                    st.caption(f"🕐 {time_str}")
                except Exception:
                    pass


def display_query_results(results_df, query_used: Optional[str] = None):
    """
    Display query results in tabular format with optional query expression.
    
    Args:
        results_df: pandas DataFrame with query results
        query_used: Optional pandas query expression that was executed
    """
    if results_df is None or results_df.empty:
        st.info("No results found for this query.")
        return
    
    # Display row count
    row_count = len(results_df)
    st.markdown(f"**Results:** {row_count} rows")
    
    # Display the data table
    st.dataframe(results_df, use_container_width=True, hide_index=False)
    
    # Display query used in an expander (commented out)
    # if query_used:
    #     with st.expander("🔍 View Query Expression"):
    #         st.code(query_used, language='python')


def main():
    """Main function for the metadata query chat interface page."""
    # Load CSS
    load_css("assets/execute_query.css")
    
    # Initialize session state
    initialize_session_state()
    
    # Validate configuration on startup
    config_valid, config_error, error_type = validate_startup_configuration()
    
    if not config_valid:
        # Display error page with setup instructions
        st.error("⚠️ **Configuration Error**")
        st.error(config_error)
        
        st.markdown("---")
        st.markdown("### 🔧 Setup Instructions")
        
        st.markdown("""
        The metadata query chat cannot load because the application is not properly configured.
        
        To set up the Metadata Query Chat, please follow these steps:
        
        1. **Create a `.env` file** in the project root directory (same location as `Welcome.py`)
        
        2. **Add the following metadata configuration** to your `.env` file:
        
        ```
        # AWS Credentials (if not already present)
        AWS_ACCESS_KEY_ID=your_access_key_here
        AWS_SECRET_ACCESS_KEY=your_secret_key_here
        AWS_REGION=us-east-1
        
        # Bedrock Configuration (if not already present)
        BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-5-v2
        
        # Metadata Configuration
        METADATA_S3_BUCKET=your-bucket-name
        METADATA_S3_KEY=optimized_metadata_full_v10.json
        
        # Optional Parameters (with defaults)
        MAX_RESULT_ROWS=10000
        QUERY_TIMEOUT_SECONDS=30
        ```
        
        3. **Fill in your configuration:**
           - `METADATA_S3_BUCKET`: S3 bucket containing your metadata JSON file
           - `METADATA_S3_KEY`: Path to the JSON file in the bucket
        
        4. **Save the file** and restart the application
        
        ---
        
        **Need help?**
        - Ensure your AWS credentials have permissions for S3 access (`s3:GetObject`)
        - Verify that your metadata file exists at the specified S3 location
        - Check that the JSON file is properly formatted
        """)
        
        # Provide link back to home
        st.info("👈 Return to the **Home** page using the sidebar navigation")
        
        return  # Prevent chat interface from loading
    
    # Display header
    st.title("📊 Query Metadata with AI")
    st.markdown("Ask questions about your metadata in natural language. I'll convert your questions into queries and show you the results.")
    st.markdown("---")
    
    # Check for configuration errors during initialization
    if st.session_state.config_error:
        st.error("⚠️ **Configuration Error**")
        st.error(st.session_state.config_error)
        st.info("""
        **Setup Instructions:**
        1. Ensure your `.env` file includes metadata configuration
        2. Add `METADATA_S3_BUCKET` and `METADATA_S3_KEY` variables
        3. Restart the application
        """)
        return
    
    # Load metadata from S3
    if not st.session_state.metadata_loaded:
        with st.spinner("📥 Loading metadata from S3..."):
            success = load_metadata_from_s3()
            
            if not success:
                display_error('s3', st.session_state.metadata_error, show_retry=False)
                st.info("Please check your configuration and refresh the page to try again.")
                return
    
    # Initialize query handler
    if not st.session_state.initialized:
        if not initialize_query_handler():
            st.error("Failed to initialize query handler")
            if st.session_state.config_error:
                st.error(st.session_state.config_error)
            return
    
    # Display metadata info in sidebar
    if st.session_state.metadata_info:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Metadata Info")
        st.sidebar.metric("Total Rows", st.session_state.metadata_info.get('row_count', 0))
        st.sidebar.metric("Columns", st.session_state.metadata_info.get('column_count', 0))
        
        with st.sidebar.expander("📋 Available Columns"):
            columns = st.session_state.metadata_info.get('columns', [])
            for col in columns:
                st.text(f"• {col}")
    
    # Display chat messages in a scrollable container
    display_chat_messages()
    
    # Display suggestion questions if available
    if st.session_state.suggestions and len(st.session_state.messages) > 0:
        st.markdown("### 💡 Suggested Questions")
        cols = st.columns(2)
        for idx, suggestion in enumerate(st.session_state.suggestions):
            col_idx = idx % 2
            with cols[col_idx]:
                if st.button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
                    # Set the suggestion as user input
                    st.session_state.selected_suggestion = suggestion
                    st.rerun()
    
    # Check if a suggestion was selected
    user_input = None
    if 'selected_suggestion' in st.session_state and st.session_state.selected_suggestion:
        user_input = st.session_state.selected_suggestion
        st.session_state.selected_suggestion = None
    else:
        # Chat input
        user_input = st.chat_input("Ask a question about the metadata...")
    
    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
            st.caption(f"🕐 {datetime.now().strftime('%I:%M %p')}")
        
        # Add user message to history
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Clear previous suggestions when user sends a new message
        st.session_state.suggestions = []
        
        # Show loading spinner with status while processing
        with st.spinner("🤔 Processing your query..."):
            # Add a placeholder for status updates
            status_placeholder = st.empty()
            status_placeholder.info("🔍 Generating query...")
            
            try:
                # Update status
                status_placeholder.info("⚙️ Executing query...")
                
                # Process query through MetadataQueryHandler
                response = st.session_state.query_handler.process_query(
                    user_query=user_input,
                    chat_history=st.session_state.messages[:-1]  # Exclude the just-added user message
                )
                
                # Update status
                status_placeholder.info("✍️ Generating summary...")
                
                # Clear status placeholder
                status_placeholder.empty()
                
                # Check for errors in response
                if not response.get('success'):
                    error_message = response.get('error', 'Unknown error')
                    
                    # Determine error type based on error message
                    if 'column' in error_message.lower():
                        error_type = 'query'
                    elif 'syntax' in error_message.lower():
                        error_type = 'query'
                    elif 'data file' in error_message.lower() or 'csv' in error_message.lower():
                        error_type = 's3'
                    else:
                        error_type = 'general'
                    
                    # Display error with contextual information
                    display_error(error_type, error_message, show_retry=True)
                    
                    logger.info(f"Query processing failed: {error_message}")
                else:
                    # Display AI response with summary and results
                    summary = response.get('summary', '')
                    results_df = response.get('results')
                    query_used = response.get('query_used', '')
                    row_count = response.get('row_count', 0)
                    data_source = response.get('data_source', 'data')
                    
                    with st.chat_message("assistant"):
                        # Display data source info (commented out)
                        # if data_source and data_source != "metadata catalog":
                        #     st.info(f"📊 **Data Source:** {data_source}")
                        
                        # Display summary
                        st.markdown(summary, unsafe_allow_html=False)
                        
                        # Display query results
                        if results_df is not None and not results_df.empty:
                            display_query_results(results_df, query_used)
                        elif row_count == 0:
                            st.info("No results found for this query.")
                        
                        # Display timestamp
                        timestamp = response.get('timestamp')
                        if timestamp:
                            try:
                                dt = datetime.fromisoformat(timestamp)
                                time_str = dt.strftime("%I:%M %p")
                                st.caption(f"🕐 {time_str}")
                            except Exception:
                                pass
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': summary,
                        'timestamp': response.get('timestamp'),
                        'query_results': results_df,
                        'query_used': query_used,
                        'row_count': row_count,
                        'data_source': data_source
                    })
                    
                    # Generate suggestion questions after assistant response
                    if st.session_state.suggestion_generator:
                        try:
                            suggestions = st.session_state.suggestion_generator.generate_suggestions(
                                conversation_context=st.session_state.messages,
                                num_suggestions=4
                            )
                            st.session_state.suggestions = suggestions
                            logger.info(f"Generated {len(suggestions)} suggestion questions")
                        except Exception as e:
                            logger.error(f"Failed to generate suggestions: {str(e)}")
                            st.session_state.suggestions = []
                    
                    # Rerun to display the new messages
                    st.rerun()
                    
            except MetadataQueryError as e:
                # Handle query-specific errors
                logger.error(f"MetadataQueryError processing query: {str(e)}")
                display_error(e.error_type, str(e), show_retry=True)
                
            except BedrockError as e:
                # Handle Bedrock-specific errors
                logger.error(f"BedrockError processing query: {str(e)}")
                display_error(e.error_type, str(e), show_retry=True)
                
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error processing query: {str(e)}", exc_info=True)
                display_error('general', str(e), show_retry=True)
    
    # Add a clear chat button in the sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
