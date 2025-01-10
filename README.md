# Colibri RSKamal

## Overview
This repository contains my submission for the Colibri Digital Wind Turbines Data Engineering Exercise. It demonstrates the design and implementation of an end-to-end pipeline using PySpark and includes machine learning models to showcase the usability of the Data Warehouse (DWH) I have designed.

## Contents
- **`wind_turbine_pipeline.ipynb`**: A Jupyter Notebook that illustrates the pipeline in detail, with outputs and comments at every stage to explain the process and objectives.
- **`wind_turbine_pipeline.py`**: A Python script implementing the same pipeline logic in PySpark for deployment purposes, this file is parameterised and uses functions to produce a DAG.
- **`Design and Assumptions.docx`**: A detailed document explaining the design choices, assumptions, and methodology used in this project.

## Key Features
1. **Pipeline Implementation**: 
   - The pipeline processes the provided data files and overwrites Delta tables.
   - Each stage is documented in the notebook with comments explaining the logic and objectives.

2. **Data Processing**: 
   - While the current implementation processes entire data files, in a real-world scenario, this would be optimized by ingesting only new data using a `load_date` field, only new data would be processed and merged into tables across layers to ensure the pipeline is scaleable
   - Workflow orchestration tools (e.g., Azure Data Factory, Apache Airflow, Databricks Workflows) would be employed to manage incremental data ingestion and merge it across layers, improving scalability and efficiency.

3. **Scalability**: 
   - The designed workflow prioritizes scalability by adopting best practices for data processing, including incremental updates and efficient table management.
   - To enhance testability, I recommend separating each processing layer into its own dedicated job. This approach isolates potential errors, making them easier to identify and resolve. For the purposes of this exercise, however, I have consolidated all layers into a single file for simplicity and demonstration.

## Usage
- Open the `wind_turbine_pipeline.ipynb` file to explore the pipeline and view the intermediate outputs at each stage.
- Refer to `wind_turbine_pipeline.py` for a PySpark implementation without intermediate outputs
- To run this notebook or `.py` file, update the `bronze_df` file path to the directory containing your `.csv` files, you may have to comment out saves to Delta tables if you want to maintain your existing table structure on you Databricks cluster.
- For a detailed explanation of the project, including design rationale and assumptions, consult the `Design and Assumptions.docx` file.

## Notes
- The pipeline overwrites Delta tables in the current setup for simplicity.
- Incremental data processing and workflow orchestration are recommended for real-world deployment.
