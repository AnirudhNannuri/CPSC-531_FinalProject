# feature_engineering.py
from pyspark.sql.functions import col

def add_consumption_features(df):
    """Calculate consumption difference and add as a new feature."""
    df = df.withColumn("consumption_diff", col("new_index") - col("old_index"))
    return df
