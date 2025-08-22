# Ivy Plus MARC Analysis with PySpark - VERSION 2.0

## Overview

This project analyzes MARC bibliographic data from Ivy Plus libraries to identify unique holdings at the University of Pennsylvania Libraries. The analysis uses PySpark for distributed processing and implements sophisticated matching algorithms to determine which records are held exclusively by Penn within the consortium.

The system processes MARC data through several stages:
1. Converts MARC files to Parquet format for efficient processing
2. Normalizes and creates multiple types of match keys for cross-institutional comparison
3. Identifies Penn records not held by other Ivy Plus institutions
4. Applies conservative filtering to ensure accuracy
5. Filters results to focus on print materials
6. Generates statistical samples and reports

## Key Features

### Enhanced Matching Algorithm (VERSION 2.0)
- **Multi-Level Matching Strategy**: Implements strict, fuzzy, and work-level match keys
- **ISBN Core Extraction**: Matches both ISBN-10 and ISBN-13 variants of the same work
- **Enhanced OCLC Extraction**: Handles all variants (ocm, ocn, on prefixes) and leading zeros
- **Multi-Volume Detection**: Identifies and properly handles multi-volume sets
- **F264 Support**: Checks both F260 and F264 fields for modern publication data
- **Smart Title Normalization**: Preserves important distinctions while removing noise
- **Conservative Analysis**: Optional filtering using only standard identifiers

### Material Type Analysis
- Uses MARC Leader (FLDR) field to categorize materials
- Separates print from electronic resources
- Identifies books, serials, music, maps, and other material types
- Filters out reproduction notes (533 fields) to focus on original materials
- Removes Historical Society of Pennsylvania (HSP) records

### Additional Filtering
- **ISBN Deduplication**: When multiple records share the same ISBN, keeps only one
- **Reproduction Removal**: Filters out records with F533 (reproduction note)
- **HSP Record Removal**: Excludes Historical Society of Pennsylvania holdings

## Project Structure

```
pod-pyspark-notebook/
├── pod-processing.ipynb          # Main analysis notebook
├── ivyplus-updated-marc-pyspark.ipynb  # MARC preprocessing notebook
├── pod-processing-outputs/       # All processing outputs
│   ├── final/                   # Processed MARC files
│   ├── export/                  # Export packages
│   ├── logs/                    # Processing logs
│   ├── all_records_exploded.parquet    # Exploded dataset for analysis
│   ├── unique_penn.parquet      # Unique Penn records
│   ├── conservative_unique_penn.parquet # Conservative estimate
│   ├── conservative_unique_penn_filtered.parquet # Final filtered results
│   ├── physical_books_no_533.parquet  # Print materials
│   └── statistical_sample_*     # Sample datasets
├── pod_*/                       # Institution-specific raw data
└── hathitrust/                  # HathiTrust-related scripts
```

## Prerequisites

### System Requirements
- **Server Environment**: This notebook is configured for a high-performance server
- **RAM**: 300GB+ total system RAM (notebook configured for 260GB Spark driver memory)
- **CPU**: 12+ cores recommended
- **Storage**: 1TB+ free disk space

### Software Requirements
- **Python**: >= 3.11
- **Java**: JDK 17 (OpenJDK recommended)
- **PySpark**: Latest version
- **marctable**: For MARC to Parquet conversion

### Python Packages
```bash
pip install --upgrade pip
pip install pymarc poetry marctable pyspark
pip install fuzzywuzzy python-Levenshtein langdetect
```

## Usage

### 1. Data Preparation
Ensure your MARC data is organized in the expected structure:
- Place processed MARC files in `pod-processing-outputs/final/`
- Or use raw institution files in `pod_[institution]/file/` directories
- Include `hsp_removed_mmsid.txt` for HSP filtering

### 2. Run the Analysis
Open and run `pod-processing.ipynb` in Jupyter:
```bash
jupyter notebook pod-processing.ipynb
```

The notebook will:
1. Convert MARC files to Parquet format (if needed)
2. Process all institutions with enhanced matching
3. Create exploded dataset for analysis
4. Identify unique Penn holdings
5. Apply conservative filters
6. Generate analysis reports and samples

### 3. Output Files

Key outputs in `pod-processing-outputs/`:
- `all_records_exploded.parquet` - Exploded dataset with one row per identifier
- `unique_penn.parquet` - All Penn records not held by other institutions
- `conservative_unique_penn.parquet` - Conservative estimate (standard IDs only)
- `conservative_unique_penn_filtered.parquet` - Final filtered unique records
- `physical_books_no_533.parquet` - Unique Penn print materials
- `penn_overlap_analysis.parquet` - Detailed overlap analysis
- `match_key_validation_stats.parquet` - Match key quality metrics
- `statistical_sample_for_api_no_hsp.csv` - Sample for validation
- `sample_summary_no_hsp.json` - Summary statistics

## Processing Workflow

### Stage 1: MARC to Parquet Conversion
- Reads MARC files with maximum error recovery
- Converts to Parquet using marctable
- Preserves MARC Leader (FLDR) field
- Maintains institution-specific separation

### Stage 2: Enhanced Processing (VERSION 2.0)
- **OCLC Enhancement**: Extracts 3x more OCLC numbers with improved patterns
- **ISBN Core**: Creates core ISBN for work-level matching
- **Multi-Volume Detection**: Flags and handles multi-volume sets
- **Publication Year**: Checks both F260 and F264 fields
- **Multiple Match Keys**: Creates strict, fuzzy, and work-level keys
- **Validation**: Each match key is validated for quality

### Stage 3: Exploded Dataset Creation
- Creates comprehensive id_list with all identifiers and match keys
- Explodes dataset to one row per identifier
- Enables efficient cross-institutional comparison

### Stage 4: Uniqueness Analysis
- Groups by identifier to find overlap
- Identifies Penn records held by no other institution (in Ivy Plus libraries)
- Calculates overlap statistics by library count
- Provides conservative estimates using only standard identifiers

### Stage 5: Additional Filtering
- ISBN deduplication (one record per ISBN)
- Removes reproductions (F533 field)
- Excludes HSP records from list

### Stage 6: Material Type Analysis
- Analyzes MARC Leader for material types
- Filters for print materials only
- Generates material type distribution

### Stage 7: Sampling and Reporting
- Creates stratified sample by material type
- Generates CSV and JSON outputs
- Produces comprehensive statistics

## Performance Optimization

The notebook includes several optimizations:
- Batch processing by institution to manage memory
- Broadcast joins for small lookup tables
- Adaptive query execution
- Arrow optimization for Pandas operations
- Periodic cache clearing
- Temporary file management

## Data Quality Considerations

### Enhanced in VERSION 2.0
- Better OCLC number extraction (handles all common patterns)
- ISBN core matching reduces false negatives
- Multi-volume detection prevents false positives
- F264 support captures modern records
- Multiple match levels provide comprehensive coverage

### Conservative Analysis Options
- Can limit to standard identifiers only (ISBN, OCLC, LCCN)
- High-confidence subset available (multiple identifier matches)
- Validation statistics for match key quality

### Known Limitations
- Some cataloging edge cases may not be normalized
- Match keys depend on consistent cataloging practices
- Multi-volume matching may miss some complex sets

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```python
   # increase memory allocation:
   .config("spark.driver.memory", "XXXg")
   .config("spark.sql.shuffle.partitions", "200")
   ```

2. **marctable Command Not Found**
   - Ensure marctable is installed: `pip install marctable`
   - Check PATH configuration in the notebook
   - May need to restart kernel after installation

3. **Empty id_list Error**
   - Fixed in VERSION 2.0
   - Ensure using `add_id_list_spark_enhanced` function
   - Check that identifiers are being extracted properly

4. **Java Errors**
   - Verify Java 17 is installed: `java -version`
   - Set JAVA_HOME: `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64`

### Logging
Processing logs are saved to `pod-processing-outputs/logs/marc2parquet.log`

## Version History

### VERSION 2.0 (Current)
- Enhanced OCLC extraction
- ISBN core matching for work-level deduplication
- Multi-volume detection
- F264 field support
- Fixed id_list generation bug
- Conservative analysis options
- HSP record filtering
- ISBN deduplication

### VERSION 1.0
- Initial implementation
- Basic match key generation
- Standard identifier extraction

## Acknowledgements

This project was developed with assistance from:
- [GitHub Copilot](https://github.com/features/copilot) - AI pair programming
- [marctable](https://github.com/sul-dlss-labs/marctable) - MARC to Parquet conversion
- [POD](https://pod.stanford.edu/) - The POD Aggregator is an open source system to aggregate MARC bibliographic and holdings data for institutions that participate in the Platform for Open Data (POD) initiative.
