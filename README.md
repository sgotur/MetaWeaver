# MetaBeaver AI ChatBot

An intelligent conversational assistant powered by AWS Bedrock and Claude Sonnet 4.5 that transforms metadata into actionable insights through natural language queries.

## Overview

MetaBeaver AI ChatBot bridges natural language understanding, metadata reasoning, and data analytics into one seamless pipeline. It automatically generates intelligent queries through a large language model (LLM), executes them using Pandas to dynamically load, filter, and analyze data, and returns structured, context-aware results.

## Features

- **🤖 Natural Language Queries**: Ask questions about your metadata in plain English
- **📊 Intelligent Data Analysis**: Automatically generates and executes Pandas queries
- **💡 Smart Suggestions**: AI-powered follow-up question recommendations
- **🔒 Secure & Private**: Enterprise-grade security with AWS Bedrock
- **📈 Real-time Results**: Interactive data visualization and tabular results
- **🎨 Modern UI**: Clean, responsive interface built with Streamlit

## Architecture

```
MetaBeaver-1/
├── Home.py                          # Main landing page
├── pages/
│   └── ExecuteQuery.py              # Metadata query chat interface
├── config/
│   └── settings.py                  # Configuration management
├── utils/
│   ├── aws_client.py                # AWS Bedrock client wrapper
│   ├── csv_data_loader.py           # CSV data loader from S3
│   ├── metadata_query_handler.py    # Query processing and execution
│   ├── s3_loader.py                 # S3 metadata loader
│   └── suggestion_generator.py      # AI-powered question suggestions
├── assets/
│   ├── style.css                    # Home page styles
│   └── execute_query.css            # Query page styles
└── requirements.txt                 # Python dependencies
```

## Prerequisites

- Python 3.11 or higher
- AWS Account with Bedrock access
- AWS credentials with appropriate permissions:
  - `bedrock:InvokeModel`
  - `s3:GetObject`

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MetaWeaver-1
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # AWS Credentials
   AWS_ACCESS_KEY_ID=your_access_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_key_here
   AWS_REGION=us-east-1
   
   # Bedrock Configuration
   BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-5-v2
   KNOWLEDGE_BASE_ID=your_knowledge_base_id_here
   
   # Metadata Configuration
   METADATA_S3_BUCKET=your-bucket-name
   METADATA_S3_KEY=optimized_metadata_full_v10.json
   
   # CSV Data Configuration (optional)
   CSV_DATA_S3_BUCKET=your-bucket-name
   CSV_DATA_S3_PREFIX=Data/
   
   # Optional Parameters
   MAX_TOKENS=2048
   TEMPERATURE=0.7
   MAX_RESULT_ROWS=10000
   QUERY_TIMEOUT_SECONDS=30
   METADATA_FLATTEN_NESTED=true
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run Home.py
   ```

2. **Navigate to the application**
   
   Open your browser to `http://localhost:8501`

3. **Query your metadata**
   - Click on "Metadata Query" in the sidebar
   - Type your question in natural language
   - View results and click suggested follow-up questions

## Example Queries

- "Show me all tables in the database"
- "What columns are available in the customer table?"
- "Find all tables that contain customer information"
- "Show me the schema for the orders table"
- "Which tables have been updated in the last 30 days?"

## Configuration Options

### AWS Credentials
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: AWS region (default: us-east-1)

### Bedrock Settings
- `BEDROCK_MODEL_ID`: Model identifier (default: anthropic.claude-sonnet-4-5-v2)
- `KNOWLEDGE_BASE_ID`: Your Bedrock Knowledge Base ID
- `MAX_TOKENS`: Maximum tokens for responses (default: 2048)
- `TEMPERATURE`: Model temperature 0-1 (default: 0.7)

### Metadata Settings
- `METADATA_S3_BUCKET`: S3 bucket containing metadata JSON
- `METADATA_S3_KEY`: Path to metadata JSON file
- `METADATA_FLATTEN_NESTED`: Flatten nested JSON structures (default: true)

### Query Settings
- `MAX_RESULT_ROWS`: Maximum rows to return (default: 10000)
- `QUERY_TIMEOUT_SECONDS`: Query timeout (default: 30)

### CSV Data Settings
- `CSV_DATA_S3_BUCKET`: S3 bucket for CSV data files
- `CSV_DATA_S3_PREFIX`: Prefix/folder for CSV files (default: Data/)

## Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: AWS Bedrock (Claude Sonnet 4.5)
- **Data Processing**: Pandas
- **Cloud Storage**: AWS S3
- **Configuration**: python-dotenv
- **Language**: Python 3.11+

## Project Structure

### Core Components

**Home.py**
- Landing page with application overview
- Configuration validation
- Navigation to query interface

**pages/ExecuteQuery.py**
- Interactive chat interface
- Query processing and execution
- Result visualization
- AI-powered suggestion questions

**config/settings.py**
- Environment variable management
- Configuration validation
- Credential handling

**utils/aws_client.py**
- AWS Bedrock client wrapper
- Model invocation
- Error handling

**utils/metadata_query_handler.py**
- Natural language to Pandas query conversion
- Query execution
- Result formatting

**utils/suggestion_generator.py**
- Context-aware question generation
- Follow-up suggestion logic

**utils/s3_loader.py**
- S3 metadata file loading
- JSON parsing and flattening

**utils/csv_data_loader.py**
- Dynamic CSV data loading from S3
- Data caching and management

## Troubleshooting

### Configuration Errors
- Ensure `.env` file exists in project root
- Verify all required environment variables are set
- Check AWS credentials are valid

### Connection Issues
- Verify AWS credentials have correct permissions
- Check network connectivity to AWS services
- Ensure S3 bucket and files exist

### Query Errors
- Verify metadata file format is correct
- Check column names match your data
- Review query syntax in error details

## License

See LICENSE file for details.

## Support

For issues and questions, please check the application's error messages which provide detailed troubleshooting steps.
