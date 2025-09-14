# LLM API and Client for MARC Data Analysis

This module provides LLM (Language Learning Model) capabilities for analyzing MARC bibliographic data, specifically designed to work with the POD PySpark notebook project.

## Files

- **`llm_api.py`** - Flask-based API server providing text analysis endpoints
- **`llm_client.py`** - Client library for interacting with the API
- **`requirements.txt`** - Python dependencies

## Features

### LLM API Server (`llm_api.py`)
- **Keyword Extraction**: Extract meaningful keywords from MARC title fields
- **Publication Pattern Analysis**: Analyze publication years, languages, and subject trends
- **Duplicate Detection**: Suggest potentially duplicate records based on similarity
- **Batch Processing**: Process multiple analysis requests efficiently
- **Health Monitoring**: Health check endpoint for service monitoring

### LLM Client (`llm_client.py`)
- **Easy-to-use Interface**: Simple client for API calls with error handling
- **Retry Logic**: Automatic retry with exponential backoff
- **Validation Utilities**: MARC record validation for common issues
- **Helper Functions**: High-level analysis functions for common tasks
- **Integration Ready**: Designed to work with existing PySpark workflows

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python llm_api.py
```
The server will start on http://localhost:5000

### 3. Use the Client
```python
from llm_client import create_client, quick_analysis

# Create client
client = create_client("http://localhost:5000")

# Analyze MARC records
records = [
    {
        "F001": "123456",
        "F245_str": "Introduction to Library Science", 
        "F020_str": "9781234567890",
        "pub_year": "2023"
    }
]

# Perform analysis
analysis = quick_analysis(records, client)
print(f"Found {analysis['total_records']} records")
```

## API Endpoints

- `GET /health` - Health check
- `POST /analyze/keywords` - Extract keywords from title
- `POST /analyze/publication-patterns` - Analyze publication patterns
- `POST /analyze/duplicates` - Suggest duplicate records
- `POST /batch/process` - Batch processing

## Integration with POD Project

These modules are designed to integrate with the existing MARC analysis workflow:

1. **PySpark Processing**: Use after initial MARC data processing to add intelligence
2. **Quality Control**: Validate records and identify potential issues
3. **Deduplication**: Enhance existing match key logic with similarity analysis
4. **Reporting**: Generate insights for collection analysis

## Example Usage with POD Data

```python
# After processing MARC data with PySpark
from llm_client import MARCAnalysisHelper

helper = MARCAnalysisHelper()

# Validate processed records
validation = helper.validate_records(penn_records)
print(f"Valid records: {validation['valid_records']}")

# Analyze collection patterns  
analysis = helper.analyze_collection(penn_records)
print(f"Publication patterns: {analysis['patterns']}")

# Find potential duplicates
if len(penn_records) <= 1000:
    duplicates = helper.find_similar_records(target_record, penn_records)
    print(f"Similar records found: {len(duplicates)}")
```

## Configuration

The client supports configuration through `ClientConfig`:

```python
from llm_client import ClientConfig, LLMClient

config = ClientConfig(
    base_url="http://your-api-server:5000",
    timeout=30,
    max_retries=3,
    retry_delay=1.0
)

client = LLMClient(config)
```

## Error Handling

Both the API and client include comprehensive error handling:
- Network timeouts and retries
- Input validation
- Rate limiting protection
- Graceful degradation

## Performance Considerations

- **Batch Processing**: Use batch endpoints for multiple operations
- **Record Limits**: Duplicate analysis limited to 1000 records per request
- **Keyword Extraction**: Limited to first 100 records for performance
- **Caching**: Consider caching results for repeated analyses

## Development and Testing

Test the core functionality without starting the API server:
```bash
python -c "from llm_client import validate_marc_data; print('Client working!')"
```

For full testing with API server, ensure Flask dependencies are installed.