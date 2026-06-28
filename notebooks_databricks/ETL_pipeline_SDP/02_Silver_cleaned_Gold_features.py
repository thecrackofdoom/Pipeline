# Databricks notebook source
# DBTITLE 1,Pipeline Overview
# MAGIC %md
# MAGIC # Silver to Gold Pipeline
# MAGIC
# MAGIC ## Purpose
# MAGIC Transform cleaned credit card data into a single analytics-ready gold table for ML training.
# MAGIC
# MAGIC ## Data Flow
# MAGIC * **Input**: `end_to_end_credit_default.uci_credit_card_silver`
# MAGIC * **Output**: `end_to_end_credit_default.uci_credit_card_gold`
# MAGIC
# MAGIC ## Feature Engineering
# MAGIC 1. **Drop ID column** (not predictive)
# MAGIC 2. **One-hot encode categorical features**: SEX, EDUCATION, MARRIAGE
# MAGIC 3. **Keep all numeric features**: payment history, bill amounts, payment amounts
# MAGIC 4. **Keep target variable**: de_pay
# MAGIC
# MAGIC ## Next Steps
# MAGIC Create a separate **ML Training Notebook** that:
# MAGIC * Loads this gold table
# MAGIC * Applies model-specific preprocessing (scaling, sampling)
# MAGIC * Trains multiple models (Logistic Regression, XGBoost, etc.)
# MAGIC * Tracks experiments with MLflow



# DBTITLE 1,Import pipeline module
# Import Spark Declarative Pipelines module
from pyspark import pipelines as dp
from pyspark.sql.functions import col, when

# COMMAND ----------

# DBTITLE 1,Gold table: Feature-engineered data for ML
@dp.materialized_view(
    name="uci_credit_card_gold",
    comment="Feature-engineered credit card data ready for ML training with one-hot encoded categorical features"
)
def uci_credit_card_gold():
    # Read from silver table
    df = spark.read.table('uci_credit_card_silver')
    
    # Drop ID column (not predictive)
    df = df.drop('ID')
    
    # One-hot encode categorical features using efficient withColumns()
    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    for cat_col in categorical_cols:
        # Get unique categories
        categories = [row[0] for row in df.select(cat_col).distinct().collect()]
        categories = sorted([int(c) for c in categories])
        
        # Build dict for withColumns() - more efficient than loop with withColumn()
        new_columns = {}
        for category in categories:
            new_columns[f"{cat_col}_{category}"] = when(col(cat_col) == category, 1).otherwise(0)
        
        # Add all new columns at once
        df = df.withColumns(new_columns)
        
        # Drop original categorical column
        df = df.drop(cat_col)
    
    return df

