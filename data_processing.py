from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler


def preprocess_data(df: DataFrame) -> DataFrame:
    # List of categorical columns that need to be indexed
    categorical_features = ['counter_statue', 'counter_type']

    # Apply StringIndexer to each categorical column
    for feature in categorical_features:
        indexer = StringIndexer(inputCol=feature, outputCol=feature + "_indexed")
        df = indexer.fit(df).transform(df)
    return df


def feature_engineering(df: DataFrame) -> DataFrame:
    df = df.withColumn("index_difference", col("new_index") - col("old_index"))
    return df