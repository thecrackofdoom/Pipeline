# Databricks notebook source
# DBTITLE 1,Pipeline Overview
# MAGIC %md
# MAGIC # Bronze to Silver Data Pipeline
# MAGIC
# MAGIC ## Purpose
# MAGIC This notebook transforms raw credit card default data from the bronze layer into cleaned, standardized data ready for analysis and modeling.
# MAGIC
# MAGIC ## Data Source
# MAGIC * **Bronze Table**: `end_to_end_credit_default.uci_credit_card`
# MAGIC * **Silver Table**: `end_to_end_credit_default.uci_credit_card_silver`
# MAGIC
# MAGIC ## Transformations Applied
# MAGIC
# MAGIC ### 1. Column Renaming
# MAGIC * `default.payment.next.month` → `de_pay` (target variable)
# MAGIC * `PAY_0` → `PAY_1` (first payment status)
# MAGIC
# MAGIC ### 2. Category Consolidation
# MAGIC * **EDUCATION**: Map rare/unknown categories (0, 5, 6) to 'others' (4)
# MAGIC * **MARRIAGE**: Map unknown (0) to 'others' (3)
# MAGIC
# MAGIC ### 3. Payment Status Normalization
# MAGIC * **PAY_1 through PAY_6**: Consolidate non-delay indicators (-2, -1, 0) to 0
# MAGIC   * -2 = no consumption
# MAGIC   * -1 = pay duly
# MAGIC   * 0 = pay duly
# MAGIC   * Positive values = months of payment delay
# MAGIC
# MAGIC ### 4. Data Quality Validation
# MAGIC * Null value checks
# MAGIC * Record count validation
# MAGIC * Target distribution analysis
# MAGIC
# MAGIC ## Architecture
# MAGIC * Uses **PySpark** for distributed processing (scales beyond single-machine memory)
# MAGIC * Writes to **Delta Lake** format (ACID transactions, time travel, schema evolution)
# MAGIC * Idempotent design (safe to re-run)


# Import Spark Declarative Pipelines module
from pyspark import pipelines as dp
from pyspark.sql.functions import when, col, sum as spark_sum

@dp.materialized_view(
    name="uci_credit_card_silver",
    comment="Cleaned credit card data with consolidated categories and normalized payment statuses"
)
def uci_credit_card_silver():
    # Load bronze table
    df_bronze = spark.read.table('uci_credit_card')
    
    # Rename columns for clarity
    df_silver = df_bronze \
        .withColumnRenamed('default.payment.next.month', 'de_pay') \
        .withColumnRenamed('PAY_0', 'PAY_1')
    
    # Consolidate EDUCATION: map rare categories (0, 5, 6) to 'others' (4)
    df_silver = df_silver.withColumn(
        'EDUCATION',
        when(col('EDUCATION').isin([0, 5, 6]), 4).otherwise(col('EDUCATION'))
    )
    
    # Consolidate MARRIAGE: map unknown (0) to 'others' (3)
    df_silver = df_silver.withColumn(
        'MARRIAGE',
        when(col('MARRIAGE') == 0, 3).otherwise(col('MARRIAGE'))
    )
    
    # Normalize payment status columns: map negative values and 0 to 0 (no delay)
    # Using withColumns for better performance
    pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    pay_transformations = {
        pay_col: when(col(pay_col).isin([-2, -1, 0]), 0).otherwise(col(pay_col))
        for pay_col in pay_columns
    }
    df_silver = df_silver.withColumns(pay_transformations)
    
    return df_silver

