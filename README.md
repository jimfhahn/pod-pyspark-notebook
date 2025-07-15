# Ivy Plus MARC Analysis with PySpark

## Overview

This project analyzes MARC bibliographic data from Ivy Plus libraries to identify unique holdings at the University of Pennsylvania. The analysis uses PySpark for distributed processing and implements sophisticated matching algorithms to determine which records are held exclusively by Penn within the consortium.

The system processes MARC data through several stages:
1. Converts MARC files to Parquet format for efficient processing
2. Normalizes and creates match keys for cross-institutional comparison
3. Identifies Penn records not held by other Ivy Plus institutions
4. Filters results to focus on print materials
5. Generates statistical samples and reports

## Key Features

### Enhanced Matching Algorithm
- **ISBN/LCCN Matching**: Normalizes standard identifiers for consistent matching across catalogs
- **Smart Match Keys**: Creates composite keys from normalized title, edition, and publication data
- **Match Key Validation**: Identifies and flags potentially problematic matches
- **Combined Approach**: Uses both standard identifiers and bibliographic match keys for comprehensive coverage

### Material Type Analysis
- Uses MARC Leader (FLDR) field to categorize materials
- Separates print from electronic resources
- Identifies books, serials, music, maps, and other material types
- Filters out reproduction notes (533 fields) to focus on original materials

## Project Structure

```
pod-pyspark-notebook/
├── pod-processing.ipynb          # Main analysis notebook
├── ivyplus-updated-marc-pyspark.ipynb  # MARC preprocessing notebook
├── pod-processing-outputs/       # All processing outputs
│   ├── final/                   # Processed MARC files
│   ├── export/                  # Export packages
│   ├── logs/                    # Processing logs
│   ├── unique_penn.parquet      # Unique Penn records
│   ├── physical_books_no_533.parquet  # Print materials
│   └── statistical_sample_*     # Sample datasets
├── pod_*/                       # Institution-specific raw data
└── hathitrust/                  # HathiTrust-related scripts
```

## Prerequisites

### Software Requirements
- **Python**: >= 3.11
- **Java**: JDK 8 or 11 (required for PySpark)
- **PySpark**: Latest version
- **marctable**: For MARC to Parquet conversion

### Storage Requirements
- Minimum: 1TB free disk space for initial processing


### Python Packages
```bash
pip install --upgrade pip
pip install pymarc poetry marctable pyspark
pip install fuzzywuzzy python-Levenshtein langdetect
```

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM for processing large datasets
- The notebook is configured to use optimized Spark settings for local processing

## Usage

### 1. Data Preparation
First, ensure your MARC data is organized in the expected structure:
- Place institution MARC files in `pod_[institution]/file/` directories
- Or use the preprocessing notebook to prepare data

### 2. Run the Analysis
Open and run `pod-processing.ipynb` in Jupyter:
```bash
jupyter notebook pod-processing.ipynb
```

The notebook will:
1. Convert MARC files to Parquet format (first run only)
2. Process all institution data to identify unique Penn holdings
3. Generate analysis reports and samples

### 3. Output Files

Key outputs in `pod-processing-outputs/`:
- `unique_penn.parquet` - All Penn records not held by other institutions
- `physical_books_no_533.parquet` - Unique Penn print materials
- `penn_overlap_analysis.parquet` - Detailed overlap analysis
- `statistical_sample_for_api_no_hsp.csv` - Sample for validation
- `sample_summary_no_hsp.json` - Summary statistics

## Processing Workflow

### Stage 1: MARC to Parquet Conversion
- Reads MARC files with error recovery
- Converts to Parquet using marctable
- Includes MARC Leader field for material type identification
- Maintains institution-specific separation

### Stage 2: Normalization and Match Key Creation
- Normalizes ISBNs (handles ISBN-10 and ISBN-13)
- Standardizes LCCNs
- Creates match keys from:
  - Title (removes articles, normalizes)
  - Edition (handles spelled-out numbers)
  - Publication year extraction
- Validates match key quality

### Stage 3: Uniqueness Analysis
- Compares Penn records against all other institutions
- Uses both standard identifiers and match keys
- Identifies records held only by Penn
- Calculates overlap statistics

### Stage 4: Material Type Filtering
- Analyzes MARC Leader to determine material types
- Filters for print materials (books, serials, music, maps)
- Excludes electronic resources and reproductions
- Generates material type statistics

### Stage 5: Sampling and Reporting
- Creates stratified sample by material type
- Generates CSV for human review
- Produces JSON summary with statistics
- Saves all intermediate results for further analysis

## Performance Optimization

The notebook includes several optimizations:
- Uses Spark SQL functions instead of Python UDFs
- Implements adaptive query execution
- Configures appropriate memory allocation
- Enables Arrow optimization for Pandas conversion
- Uses broadcasting for efficient joins

## Data Quality Considerations

### Match Key Quality
The system validates match keys to identify:
- Keys that are too short (< 5 characters)
- Generic keys (e.g., "book_2023")
- Missing data issues

### Known Limitations
- Edition matching may not catch all variations
- Publication year extraction handles years 1000-2099
- Some cataloging variations may not be normalized

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `spark.executor.memory` in configuration
   - Process fewer institutions at once
   - Increase system RAM

2. **marctable Command Not Found**
   - Ensure marctable is installed: `pip install marctable`
   - Check PATH configuration in the notebook
   - May need to restart kernel after installation

3. **No Parquet Files Found**
   - Run the MARC conversion cell first
   - Check that MARC files exist in expected locations
   - Verify file permissions

### Logging
Processing logs are saved to `pod-processing-outputs/logs/marc2parquet.log`

## Future Enhancements

Potential improvements under consideration:
- Machine learning for better match key generation
- Fuzzy matching for title variations
- Integration with external authority files
- Real-time API validation of results

## Acknowledgements

This project was developed with assistance from:
- [GitHub Copilot](https://github.com/features/copilot) - AI pair programming
- [marctable](https://github.com/sul-dlss-labs/marctable) - MARC to Parquet conversion
- [POD](https://pod.stanford.edu/) - A data lake of MARC from the Ivy Plus Libraries Confederation.

