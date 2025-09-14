"""
LLM Client Library for MARC Data Analysis
Provides easy-to-use client interface for the LLM API server.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """Configuration for LLM API client"""
    base_url: str = "http://localhost:5000"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True

@dataclass
class APIResponse:
    """Wrapper for API responses with common fields"""
    success: bool
    result: Any
    message: str
    timestamp: str
    processing_time: float
    status_code: int

class LLMClientError(Exception):
    """Custom exception for LLM client errors"""
    pass

class LLMClient:
    """Client for interacting with MARC LLM API"""
    
    def __init__(self, config: Optional[ClientConfig] = None):
        """Initialize the LLM client with configuration"""
        self.config = config or ClientConfig()
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'MARC-LLM-Client/1.0'
        })
        
        logger.info(f"LLM Client initialized with base URL: {self.config.base_url}")
    
    def _make_request(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request to API with retry logic"""
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(
                        url, 
                        timeout=self.config.timeout,
                        verify=self.config.verify_ssl
                    )
                elif method.upper() == "POST":
                    response = self.session.post(
                        url,
                        json=data,
                        timeout=self.config.timeout,
                        verify=self.config.verify_ssl
                    )
                else:
                    raise LLMClientError(f"Unsupported HTTP method: {method}")
                
                # Parse response
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = {
                        "success": False,
                        "message": "Invalid JSON response",
                        "result": None,
                        "timestamp": datetime.now().isoformat(),
                        "processing_time": 0.0
                    }
                
                return APIResponse(
                    success=response_data.get('success', False),
                    result=response_data.get('result'),
                    message=response_data.get('message', ''),
                    timestamp=response_data.get('timestamp', ''),
                    processing_time=response_data.get('processing_time', 0.0),
                    status_code=response.status_code
                )
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise LLMClientError(f"Request failed after {self.config.max_retries} attempts: {str(e)}")
    
    def health_check(self) -> APIResponse:
        """Check if the API server is healthy"""
        return self._make_request("health", "GET")
    
    def extract_keywords(self, title: str) -> APIResponse:
        """Extract keywords from a title string"""
        if not title:
            raise LLMClientError("Title cannot be empty")
        
        data = {"title": title}
        return self._make_request("analyze/keywords", "POST", data)
    
    def analyze_publication_patterns(self, records: List[Dict]) -> APIResponse:
        """Analyze publication patterns in MARC records"""
        if not records:
            raise LLMClientError("Records list cannot be empty")
        
        if not isinstance(records, list):
            raise LLMClientError("Records must be a list of dictionaries")
        
        data = {"records": records}
        return self._make_request("analyze/publication-patterns", "POST", data)
    
    def suggest_duplicates(self, records: List[Dict], threshold: float = 0.8) -> APIResponse:
        """Suggest potential duplicate records"""
        if not records:
            raise LLMClientError("Records list cannot be empty")
        
        if not isinstance(records, list):
            raise LLMClientError("Records must be a list of dictionaries")
        
        if len(records) > 1000:
            raise LLMClientError("Too many records. Maximum 1000 allowed.")
        
        if not 0.0 <= threshold <= 1.0:
            raise LLMClientError("Threshold must be between 0.0 and 1.0")
        
        data = {
            "records": records,
            "threshold": threshold
        }
        return self._make_request("analyze/duplicates", "POST", data)
    
    def batch_process(self, requests_batch: List[Dict]) -> APIResponse:
        """Process multiple analysis requests in a batch"""
        if not requests_batch:
            raise LLMClientError("Batch requests list cannot be empty")
        
        if not isinstance(requests_batch, list):
            raise LLMClientError("Batch requests must be a list of dictionaries")
        
        # Validate batch request format
        for i, req in enumerate(requests_batch):
            if not isinstance(req, dict):
                raise LLMClientError(f"Request {i} must be a dictionary")
            
            if 'operation' not in req:
                raise LLMClientError(f"Request {i} missing 'operation' field")
            
            valid_operations = ['extract_keywords', 'analyze_patterns', 'suggest_duplicates']
            if req['operation'] not in valid_operations:
                raise LLMClientError(f"Request {i} has invalid operation. Must be one of: {valid_operations}")
        
        data = {"requests": requests_batch}
        return self._make_request("batch/process", "POST", data)

class MARCAnalysisHelper:
    """High-level helper class for common MARC analysis tasks"""
    
    def __init__(self, client: Optional[LLMClient] = None):
        """Initialize with an LLM client"""
        self.client = client or LLMClient()
    
    def analyze_collection(self, records: List[Dict], 
                          include_duplicates: bool = True,
                          duplicate_threshold: float = 0.8) -> Dict:
        """Perform comprehensive analysis of a MARC collection"""
        results = {
            "total_records": len(records),
            "analysis_timestamp": datetime.now().isoformat(),
            "patterns": None,
            "duplicates": None,
            "keywords": {},
            "errors": []
        }
        
        try:
            # Analyze publication patterns
            logger.info("Analyzing publication patterns...")
            patterns_response = self.client.analyze_publication_patterns(records)
            if patterns_response.success:
                results["patterns"] = patterns_response.result
            else:
                results["errors"].append(f"Pattern analysis failed: {patterns_response.message}")
            
            # Suggest duplicates if requested
            if include_duplicates and len(records) <= 1000:
                logger.info("Analyzing potential duplicates...")
                duplicates_response = self.client.suggest_duplicates(records, duplicate_threshold)
                if duplicates_response.success:
                    results["duplicates"] = duplicates_response.result
                else:
                    results["errors"].append(f"Duplicate analysis failed: {duplicates_response.message}")
            elif len(records) > 1000:
                results["errors"].append("Too many records for duplicate analysis (max 1000)")
            
            # Extract keywords from titles
            logger.info("Extracting keywords from titles...")
            keyword_counts = {}
            for i, record in enumerate(records[:100]):  # Limit to first 100 for performance
                title = record.get('F245_str', '')
                if title:
                    try:
                        keywords_response = self.client.extract_keywords(title)
                        if keywords_response.success:
                            keywords = keywords_response.result.get('keywords', [])
                            for keyword in keywords:
                                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                    except Exception as e:
                        results["errors"].append(f"Keyword extraction failed for record {i}: {str(e)}")
            
            # Sort keywords by frequency
            results["keywords"] = dict(sorted(keyword_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:50])
            
        except Exception as e:
            results["errors"].append(f"Collection analysis failed: {str(e)}")
        
        return results
    
    def find_similar_records(self, target_record: Dict, 
                           candidate_records: List[Dict],
                           threshold: float = 0.8) -> List[Dict]:
        """Find records similar to a target record"""
        if not target_record or not candidate_records:
            return []
        
        # Combine target with candidates for duplicate analysis
        all_records = [target_record] + candidate_records
        
        try:
            response = self.client.suggest_duplicates(all_records, threshold)
            if response.success:
                suggestions = response.result.get('suggestions', [])
                
                # Filter to only include suggestions involving the target record (index 0)
                target_id = target_record.get('F001', 'record_0')
                similar_records = []
                
                for suggestion in suggestions:
                    if (suggestion['record1_id'] == target_id or 
                        suggestion['record2_id'] == target_id):
                        similar_records.append(suggestion)
                
                return similar_records
            
        except Exception as e:
            logger.error(f"Error finding similar records: {str(e)}")
        
        return []
    
    def validate_records(self, records: List[Dict]) -> Dict:
        """Validate MARC records for common issues"""
        validation_results = {
            "total_records": len(records),
            "valid_records": 0,
            "issues": [],
            "statistics": {
                "missing_title": 0,
                "missing_identifiers": 0,
                "duplicate_ids": 0,
                "invalid_years": 0
            }
        }
        
        seen_ids = set()
        
        for i, record in enumerate(records):
            record_issues = []
            
            # Check for required fields
            if not record.get('F245_str'):
                record_issues.append("Missing title (F245)")
                validation_results["statistics"]["missing_title"] += 1
            
            # Check for identifiers
            has_identifier = any([
                record.get('F020_str'),  # ISBN
                record.get('F010_str'),  # LCCN
                record.get('F035_str'),  # OCLC
                record.get('F001')       # Control number
            ])
            
            if not has_identifier:
                record_issues.append("Missing identifiers")
                validation_results["statistics"]["missing_identifiers"] += 1
            
            # Check for duplicate control numbers
            control_num = record.get('F001')
            if control_num:
                if control_num in seen_ids:
                    record_issues.append(f"Duplicate control number: {control_num}")
                    validation_results["statistics"]["duplicate_ids"] += 1
                else:
                    seen_ids.add(control_num)
            
            # Check publication year validity
            pub_year = record.get('pub_year')
            if pub_year:
                try:
                    year = int(pub_year)
                    current_year = datetime.now().year
                    if year < 1000 or year > current_year + 10:
                        record_issues.append(f"Invalid publication year: {year}")
                        validation_results["statistics"]["invalid_years"] += 1
                except (ValueError, TypeError):
                    record_issues.append(f"Invalid publication year format: {pub_year}")
                    validation_results["statistics"]["invalid_years"] += 1
            
            if record_issues:
                validation_results["issues"].append({
                    "record_index": i,
                    "control_number": control_num,
                    "issues": record_issues
                })
            else:
                validation_results["valid_records"] += 1
        
        return validation_results

# Convenience functions for common operations
def create_client(base_url: str = "http://localhost:5000", 
                 timeout: int = 30) -> LLMClient:
    """Create a configured LLM client"""
    config = ClientConfig(base_url=base_url, timeout=timeout)
    return LLMClient(config)

def quick_analysis(records: List[Dict], 
                  client: Optional[LLMClient] = None) -> Dict:
    """Perform quick analysis of MARC records"""
    helper = MARCAnalysisHelper(client)
    return helper.analyze_collection(records, include_duplicates=True)

def validate_marc_data(records: List[Dict]) -> Dict:
    """Validate MARC records for common issues"""
    helper = MARCAnalysisHelper()
    return helper.validate_records(records)

if __name__ == "__main__":
    # Example usage
    print("MARC LLM Client Library")
    print("======================")
    
    # Create client
    client = create_client()
    
    # Test health check
    try:
        health = client.health_check()
        if health.success:
            print("✅ API server is healthy")
        else:
            print("❌ API server health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to API server: {str(e)}")
    
    # Example record for testing
    sample_records = [
        {
            "F001": "123456",
            "F245_str": "Introduction to Library Science",
            "F020_str": "9781234567890",
            "pub_year": "2023"
        },
        {
            "F001": "789012",
            "F245_str": "Advanced Cataloging Techniques",
            "F010_str": "2023123456",
            "pub_year": "2023"
        }
    ]
    
    print(f"\nExample analysis with {len(sample_records)} sample records:")
    try:
        analysis = quick_analysis(sample_records, client)
        print(f"Total records: {analysis['total_records']}")
        print(f"Errors: {len(analysis['errors'])}")
        if analysis['patterns']:
            print(f"Publication years found: {len(analysis['patterns']['publication_years'])}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")