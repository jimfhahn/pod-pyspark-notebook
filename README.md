# [Transactional POD using PySpark](https://github.com/jimfhahn/pod-pyspark-notbook/blob/main/pod-processing.ipynb)

## Overview
The POD environment is designed as a data lake. To facilitate data processing, the MARC data is first loaded into Parquet format and then into Spark for processing using PySpark SQL.

## Steps
### Data Loading: 
The MARC data is loaded into Parquet format using marctable. Parquet is a columnar storage file format optimized for use with big data processing frameworks.
### Data Processing: 
The Parquet files are then loaded into Spark. Using PySpark SQL, various data processing tasks are performed to transform and analyze the data.
### [Post-Processing.ipynb](https://github.com/jimfhahn/pod-pyspark-notebook/blob/main/post-processing.ipynb):
The primary goal of this notebook is to confirm that an item is **not** held by any BorrowDirect institution besides Penn. Only items with only Penn holdings are included in the final report.
## Requirements
### Python: 
Ensure you have Python => 3.11 installed.
### PySpark: 
Install PySpark to enable Spark processing.
### Marctable:
[marctable](https://github.com/sul-dlss-labs/marctable) converts MARC bibliographic data (in transmission format or MARCXML) into tabular formats like CSV and Parquet. 
### Parquet: 
Use the Parquet file format for efficient data storage and retrieval.
### Selenium web driver:
To verfiy uniqueness of holdings, we use Selenium to scrape BorrowDirect holdings data based on BorrowDirect id.

