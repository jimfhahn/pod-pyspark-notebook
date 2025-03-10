# [Transactional POD using PySpark](https://github.com/jimfhahn/pod-pyspark-notbook/blob/main/pod-processing.ipynb)

## Overview
The POD environment is designed as a data lake. To facilitate data processing, the MARC data is first loaded into Parquet format and then into Spark for processing using PySpark SQL.

## Steps
### Data Loading: 
The MARC data is loaded into Parquet format using marctable. Parquet is a columnar storage file format optimized for use with big data processing frameworks.
### Data Processing: 
The Parquet files are then loaded into Spark. Using PySpark SQL, various data processing tasks are performed to transform and analyze the data.

## Requirements
### Python: 
Ensure you have Python => 3.11 installed.
### PySpark: 
Install PySpark to enable Spark processing.
### Marctable:
[marctable](https://github.com/sul-dlss-labs/marctable) converts MARC bibliographic data (in transmission format or MARCXML) into tabular formats like CSV and Parquet. 
### Parquet: 
Use the Parquet file format for efficient data storage and retrieval.

