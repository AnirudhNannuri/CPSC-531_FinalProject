# model.py
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when


def add_fraud_label(df: DataFrame) -> DataFrame:
    df = df.withColumn("label",
                       when(
                           (col("consommation_level_1") > 10000) |
                           (col("counter_statue") == "not working" & col("consommation_level_1") > 0),
                           1).otherwise(0))
    return df


def build_and_evaluate_model(df: DataFrame) -> Pipeline:
    df = add_fraud_label(df)  # Add a label column based on specific criteria

    indexers = [StringIndexer(inputCol=c, outputCol=c + "_indexed") for c in ["counter_statue", "counter_type"]]
    assembler = VectorAssembler(
        inputCols=[c + "_indexed" for c in ["counter_statue", "counter_type"]] + ["consommation_level_1"],
        outputCol="features"
    )
    classifier = RandomForestClassifier(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=indexers + [assembler, classifier])

    model = pipeline.fit(df)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(model.transform(df))
    print(f"Training set AUC: {auc}")
    return model


def make_predictions(model: Pipeline, test_df: DataFrame) -> DataFrame:
    test_df = add_fraud_label(test_df)  # Ensure test data also has labels for evaluation
    return model.transform(test_df)
