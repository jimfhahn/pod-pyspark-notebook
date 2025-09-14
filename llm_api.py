"""
LLM API Server for MARC Data Analysis
Provides text analysis and bibliographic data processing capabilities.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@dataclass
class LLMResponse:
    """Standard response format for LLM API calls"""
    success: bool
    result: Any
    message: str
    timestamp: str
    processing_time: float

class MARCTextAnalyzer:
    """Text analysis utilities for MARC bibliographic data"""
    
    def __init__(self):
        self.common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'
        }
    
    def extract_title_keywords(self, title: str) -> List[str]:
        """Extract meaningful keywords from title field"""
        if not title:
            return []
        
        # Clean and normalize title
        title = re.sub(r'[^\w\s]', ' ', title.lower())
        words = title.split()
        
        # Filter out common words and short words
        keywords = [word for word in words 
                   if len(word) > 2 and word not in self.common_words]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def analyze_publication_patterns(self, records: List[Dict]) -> Dict:
        """Analyze publication patterns in MARC records"""
        if not records:
            return {"error": "No records provided"}
        
        analysis = {
            "total_records": len(records),
            "publication_years": {},
            "languages": {},
            "publishers": {},
            "subject_keywords": {}
        }
        
        for record in records:
            # Publication year analysis
            pub_year = record.get('pub_year', 'Unknown')
            analysis["publication_years"][pub_year] = \
                analysis["publication_years"].get(pub_year, 0) + 1
            
            # Language analysis
            language = record.get('language', 'Unknown')
            analysis["languages"][language] = \
                analysis["languages"].get(language, 0) + 1
            
            # Title keyword analysis
            title = record.get('F245_str', '')
            keywords = self.extract_title_keywords(title)
            for keyword in keywords:
                analysis["subject_keywords"][keyword] = \
                    analysis["subject_keywords"].get(keyword, 0) + 1
        
        # Sort by frequency
        for key in ["publication_years", "languages", "subject_keywords"]:
            analysis[key] = dict(sorted(analysis[key].items(), 
                                     key=lambda x: x[1], reverse=True)[:20])
        
        return analysis
    
    def suggest_duplicates(self, records: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """Suggest potential duplicate records based on similarity"""
        suggestions = []
        
        for i, record1 in enumerate(records):
            for j, record2 in enumerate(records[i+1:], i+1):
                similarity = self._calculate_similarity(record1, record2)
                if similarity >= threshold:
                    suggestions.append({
                        "record1_id": record1.get('F001', f'record_{i}'),
                        "record2_id": record2.get('F001', f'record_{j}'),
                        "similarity_score": similarity,
                        "matching_fields": self._get_matching_fields(record1, record2)
                    })
        
        return sorted(suggestions, key=lambda x: x['similarity_score'], reverse=True)
    
    def _calculate_similarity(self, record1: Dict, record2: Dict) -> float:
        """Calculate similarity score between two MARC records"""
        title1 = record1.get('F245_str', '').lower()
        title2 = record2.get('F245_str', '').lower()
        
        # Simple Jaccard similarity for titles
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 and not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_matching_fields(self, record1: Dict, record2: Dict) -> List[str]:
        """Get list of matching fields between two records"""
        matching = []
        
        # Check common fields
        fields_to_check = ['F020_str', 'F010_str', 'F035_str', 'pub_year']
        for field in fields_to_check:
            val1 = record1.get(field)
            val2 = record2.get(field)
            if val1 and val2 and val1 == val2:
                matching.append(field)
        
        return matching

# Initialize analyzer
analyzer = MARCTextAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MARC LLM API"
    })

@app.route('/analyze/keywords', methods=['POST'])
def extract_keywords():
    """Extract keywords from title text"""
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        title = data.get('title', '')
        
        if not title:
            return jsonify({
                "success": False,
                "message": "Title is required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        keywords = analyzer.extract_title_keywords(title)
        
        response = LLMResponse(
            success=True,
            result={"keywords": keywords},
            message="Keywords extracted successfully",
            timestamp=datetime.now().isoformat(),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        return jsonify(response.__dict__)
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analyze/publication-patterns', methods=['POST'])
def analyze_publication_patterns():
    """Analyze publication patterns in MARC records"""
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        records = data.get('records', [])
        
        if not records:
            return jsonify({
                "success": False,
                "message": "Records are required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        analysis = analyzer.analyze_publication_patterns(records)
        
        response = LLMResponse(
            success=True,
            result=analysis,
            message="Publication patterns analyzed successfully",
            timestamp=datetime.now().isoformat(),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        return jsonify(response.__dict__)
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/analyze/duplicates', methods=['POST'])
def suggest_duplicates():
    """Suggest potential duplicate records"""
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        records = data.get('records', [])
        threshold = data.get('threshold', 0.8)
        
        if not records:
            return jsonify({
                "success": False,
                "message": "Records are required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        if len(records) > 1000:
            return jsonify({
                "success": False,
                "message": "Too many records. Maximum 1000 allowed.",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        suggestions = analyzer.suggest_duplicates(records, threshold)
        
        response = LLMResponse(
            success=True,
            result={
                "suggestions": suggestions,
                "total_suggestions": len(suggestions),
                "threshold_used": threshold
            },
            message="Duplicate suggestions generated successfully",
            timestamp=datetime.now().isoformat(),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        return jsonify(response.__dict__)
        
    except Exception as e:
        logger.error(f"Error suggesting duplicates: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/batch/process', methods=['POST'])
def batch_process():
    """Process multiple analysis requests in a batch"""
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        requests_batch = data.get('requests', [])
        
        if not requests_batch:
            return jsonify({
                "success": False,
                "message": "Batch requests are required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        results = []
        for i, req in enumerate(requests_batch):
            operation = req.get('operation')
            params = req.get('parameters', {})
            
            try:
                if operation == 'extract_keywords':
                    result = analyzer.extract_title_keywords(params.get('title', ''))
                elif operation == 'analyze_patterns':
                    result = analyzer.analyze_publication_patterns(params.get('records', []))
                elif operation == 'suggest_duplicates':
                    result = analyzer.suggest_duplicates(
                        params.get('records', []), 
                        params.get('threshold', 0.8)
                    )
                else:
                    result = {"error": f"Unknown operation: {operation}"}
                
                results.append({
                    "request_id": i,
                    "operation": operation,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "request_id": i,
                    "operation": operation,
                    "success": False,
                    "error": str(e)
                })
        
        response = LLMResponse(
            success=True,
            result={
                "batch_results": results,
                "total_processed": len(results),
                "successful": len([r for r in results if r['success']])
            },
            message="Batch processing completed",
            timestamp=datetime.now().isoformat(),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        return jsonify(response.__dict__)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting MARC LLM API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)