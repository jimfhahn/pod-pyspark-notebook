#!/usr/bin/env python3
"""
Example integration of LLM modules with POD MARC data processing
This script demonstrates how to use the new LLM capabilities with existing data.
"""

import sys
import json
from typing import List, Dict

def simulate_marc_processing():
    """Simulate MARC records from the POD processing pipeline"""
    
    # These would typically come from the PySpark processing in pod-processing.ipynb
    sample_penn_records = [
        {
            "F001": "penn001",
            "F245_str": "Digital Humanities and Library Science: A Comprehensive Guide",
            "F020_str": "9780123456789",
            "F260_str": "Philadelphia : University of Pennsylvania Press, 2023",
            "F264_str": "Philadelphia : University of Pennsylvania Press, 2023",
            "pub_year": "2023",
            "institution": "PENN",
            "match_key": "digital_humanities_library_science_comprehensive_guide_2023",
            "isbn_core": "0123456789"
        },
        {
            "F001": "penn002",
            "F245_str": "Modern Cataloging Practices in Academic Libraries",
            "F010_str": "2023001234",
            "F260_str": "Boston : Academic Libraries Press, 2022", 
            "pub_year": "2022",
            "institution": "PENN",
            "match_key": "modern_cataloging_practices_academic_libraries_2022",
            "oclc_num": "1234567890"
        },
        {
            "F001": "penn003",
            "F245_str": "Introduction to Information Science and Technology",
            "F035_str": "(OCoLC)987654321",
            "F264_str": "New York : Information Science Publishers, 2021",
            "pub_year": "2021", 
            "institution": "PENN",
            "match_key": "introduction_information_science_technology_2021",
            "oclc_num": "987654321"
        },
        {
            "F001": "penn004",
            "F245_str": "Digital Humanities and Library Science: A Guide",  # Similar to first record
            "F020_str": "9780987654321",
            "pub_year": "2023",
            "institution": "PENN",
            "match_key": "digital_humanities_library_science_guide_2023",
            "isbn_core": "0987654321"
        },
        {
            "F001": "penn005",
            "F245_str": "",  # Missing title - quality issue
            "F020_str": "9781111111111",
            "pub_year": "invalid",  # Invalid year - quality issue
            "institution": "PENN"
        }
    ]
    
    return sample_penn_records

def demonstrate_llm_analysis():
    """Demonstrate LLM analysis capabilities"""
    
    print("🏛️  POD PySpark + LLM Analysis Demonstration")
    print("=" * 60)
    
    # Get sample data (normally from PySpark processing)
    records = simulate_marc_processing()
    print(f"📚 Loaded {len(records)} sample PENN records from processing pipeline\n")
    
    try:
        from llm_client import (
            create_client, 
            validate_marc_data, 
            MARCAnalysisHelper,
            LLMClientError
        )
        
        # 1. Validate record quality
        print("1️⃣  RECORD QUALITY VALIDATION")
        print("-" * 30)
        
        validation = validate_marc_data(records)
        print(f"Total records: {validation['total_records']}")
        print(f"Valid records: {validation['valid_records']}")
        print(f"Records with issues: {len(validation['issues'])}")
        
        if validation['issues']:
            print("\nQuality Issues Found:")
            for issue in validation['issues'][:3]:  # Show first 3
                print(f"  • Record {issue['record_index']}: {', '.join(issue['issues'])}")
        
        print(f"\nStatistics:")
        for stat, count in validation['statistics'].items():
            if count > 0:
                print(f"  • {stat.replace('_', ' ').title()}: {count}")
        
        # 2. Analyze publication patterns  
        print(f"\n2️⃣  PUBLICATION PATTERN ANALYSIS")
        print("-" * 35)
        
        helper = MARCAnalysisHelper()
        
        # Filter to valid records for analysis
        valid_records = [r for i, r in enumerate(records) 
                        if i not in [issue['record_index'] for issue in validation['issues']]]
        
        print(f"Analyzing {len(valid_records)} valid records...")
        
        # Since API server isn't running, we'll use the helper's validation function
        # In real usage, this would call the API
        print("\n📊 Collection Overview:")
        print(f"  • Publication Years: {set(r.get('pub_year', 'Unknown') for r in valid_records)}")
        print(f"  • Institutions: {set(r.get('institution', 'Unknown') for r in valid_records)}")
        
        # Extract keywords manually (would normally use API)
        all_keywords = []
        for record in valid_records:
            title = record.get('F245_str', '')
            if title:
                # Simple keyword extraction
                words = title.lower().replace(',', '').replace(':', '').split()
                keywords = [w for w in words if len(w) > 3 and w not in 
                           ['the', 'and', 'with', 'for', 'from', 'into']]
                all_keywords.extend(keywords)
        
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n🔍 Top Keywords:")
        for keyword, count in top_keywords:
            print(f"  • {keyword}: {count} occurrences")
        
        # 3. Identify potential duplicates
        print(f"\n3️⃣  DUPLICATE DETECTION")
        print("-" * 25)
        
        # Simple similarity check (would normally use API)
        potential_duplicates = []
        for i, record1 in enumerate(valid_records):
            for j, record2 in enumerate(valid_records[i+1:], i+1):
                title1 = record1.get('F245_str', '').lower()
                title2 = record2.get('F245_str', '').lower()
                
                # Simple word-based similarity
                words1 = set(title1.split())
                words2 = set(title2.split())
                
                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.6:  # 60% similarity threshold
                        potential_duplicates.append({
                            'record1': record1['F001'],
                            'record2': record2['F001'],
                            'similarity': similarity,
                            'title1': record1.get('F245_str', ''),
                            'title2': record2.get('F245_str', '')
                        })
        
        if potential_duplicates:
            print(f"Found {len(potential_duplicates)} potential duplicate pairs:")
            for dup in potential_duplicates:
                print(f"  • {dup['record1']} ↔ {dup['record2']} ({dup['similarity']:.1%} similar)")
                print(f"    '{dup['title1'][:50]}...'")
                print(f"    '{dup['title2'][:50]}...'")
        else:
            print("No potential duplicates found")
        
        # 4. Integration recommendations
        print(f"\n4️⃣  INTEGRATION RECOMMENDATIONS")
        print("-" * 37)
        
        print("💡 To integrate LLM analysis with your PySpark workflow:")
        print("  1. After processing MARC data in pod-processing.ipynb")
        print("  2. Export records to JSON/dict format")
        print("  3. Use llm_client for quality validation and analysis")
        print("  4. Incorporate results back into PySpark DataFrame")
        
        print(f"\n📝 Example integration code:")
        print("""
# In your Jupyter notebook, after PySpark processing:
from llm_client import validate_marc_data, create_client

# Convert PySpark DataFrame to list of dicts
penn_records = penn_df.toPandas().to_dict('records')

# Validate quality
validation = validate_marc_data(penn_records)
print(f"Quality: {validation['valid_records']}/{validation['total_records']} valid")

# If API server is running, perform advanced analysis
try:
    client = create_client("http://localhost:5000")
    analysis = client.analyze_publication_patterns(penn_records)
    if analysis.success:
        patterns = analysis.result
        print(f"Publication years: {patterns['publication_years']}")
except Exception as e:
    print(f"API not available: {e}")
""")
        
        print(f"\n✅ Demonstration complete!")
        print(f"🚀 Ready to enhance your MARC analysis with LLM capabilities")
        
    except ImportError as e:
        print(f"❌ Could not import LLM modules: {e}")
        print("   Make sure llm_client.py is in the Python path")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    demonstrate_llm_analysis()