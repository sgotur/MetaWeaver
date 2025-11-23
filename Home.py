import os
import warnings
import logging

# Configure logging to suppress ScriptRunContext warnings
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')

import streamlit as st
from config.settings import load_config, ConfigurationError

# Page with title, icon, and layout settings
st.set_page_config(
    page_title="MetaBeaver AI ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS from external file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Validate configuration on startup
def validate_startup_configuration():
    """
    Validate configuration on application startup.
    Returns tuple of (is_valid, error_message)
    """
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            return False, "Configuration file '.env' is missing."
        
        # Try to load and validate configuration
        settings = load_config()
        
        # Validate AWS credentials are present
        aws_creds = settings.get_aws_credentials()
        
        # Validate Bedrock configuration
        bedrock_config = settings.get_bedrock_config()
        
        return True, None
        
    except ConfigurationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected configuration error: {str(e)}"

# Load CSS after page config
load_css("assets/style.css")

# Check configuration validity
config_valid, config_error = validate_startup_configuration()

# Display configuration error if validation failed
if not config_valid:
    st.error("⚠️ **Configuration Error**")
    st.error(config_error)
    
    st.markdown("---")
    st.markdown("### 🔧 Setup Instructions")
    
    st.markdown("""
    To set up the Boxer AI ChatBot, please follow these steps:
    
    1. **Create a `.env` file** in the project root directory (same location as `Welcome.py`)
    
    2. **Copy the template** from `.env.example` if available, or create a new file with the following structure:
    
    ```
    # AWS Credentials
    AWS_ACCESS_KEY_ID=your_access_key_here
    AWS_SECRET_ACCESS_KEY=your_secret_key_here
    AWS_REGION=us-east-1
    
    # Bedrock Configuration
    BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-5-v2
    KNOWLEDGE_BASE_ID=your_knowledge_base_id_here
    
    # Optional Parameters (with defaults)
    MAX_TOKENS=2048
    TEMPERATURE=0.7
    ```
    
    3. **Fill in your AWS credentials:**
       - `AWS_ACCESS_KEY_ID`: Your AWS access key
       - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key
       - `AWS_REGION`: AWS region where your Bedrock resources are located
    
    4. **Configure Bedrock settings:**
       - `BEDROCK_MODEL_ID`: The Bedrock model identifier (e.g., `anthropic.claude-sonnet-4-5-v2`)
       - `KNOWLEDGE_BASE_ID`: Your Bedrock Knowledge Base ID
    
    5. **Save the file** and restart the application
    
    ---
    
    **Need help?**
    - Ensure your AWS credentials have permissions for Bedrock and Knowledge Base access
    - Verify that your Knowledge Base is created and indexed in AWS Bedrock
    - Check that all required environment variables are set and not empty
    """)
    
    st.stop()  # Prevent rest of the page from loading

# Sidebar navigation menu with icons
st.sidebar.title("MetaBeaver")
st.sidebar.markdown("---")

# Navigation menu items
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.rerun()
if st.sidebar.button("📊 Metadata Query", use_container_width=True):
    st.switch_page("pages/ExecuteQuery.py")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("Meta Beaver AI ChatBot v1.0\n\nPowered by AWS Bedrock & \nClaude Sonnet 4.5")

# Display company logo and name in header
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(logo_path, width=100)
    with col2:
        st.title("Boxer AI")
        st.subheader("Intelligent Conversational Assistant")
else:
    # Fallback if logo doesn't exist
    st.title("🤖 MetaBeaver AI")
    st.subheader("Intelligent Conversational Assistant to generate metadata queries")

# Welcome message and application description
st.markdown("---")
st.write("## Welcome to MetaBeaver AI ChatBot!")

st.markdown("""
### About This Application

Welcome to the MetaBeaver AI ChatBot, your intelligent assistant powered by AWS Bedrock and Anthropic's Claude Sonnet 4.5. 
This application combines cutting-edge AI technology to provide accurate, contextually relevant responses to your questions in NLP.

#### Key Features:

- **🤖 Advanced AI Conversations**: Engage with Claude Sonnet 4.5, one of the most sophisticated language models available
- **💡 Smart Suggestions**: Receive intelligent follow-up question suggestions to explore topics deeper
- **🔒 Secure & Private**: Your conversations are secure with AWS Bedrock's enterprise-grade security

#### Getting Started:

1. Navigate to the **Execute Query** page using the sidebar menu
2. Type your question or select from suggested topics
3. Receive intelligent, context-aware responses
4. Explore related topics with AI-generated follow-up questions
""")

st.markdown("---")
st.info("💡 **Tip**: Use the navigation menu on the left to access different features of the application.")
