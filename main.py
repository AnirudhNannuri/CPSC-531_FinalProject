from pyspark.ml.tuning import ParamGridBuilder
from load_data import load_data_from_cassandra
from data_processing import preprocess_data, feature_engineering
from model import build_and_evaluate_model, make_predictions
from evaluation import evaluate_model
from visualizations import plot_correlation_matrix, target_variable
from tuning import tune_model
from pyspark.sql import SparkSession
import pandas as pd

def get_spark_session():
    spark = SparkSession.builder \
        .appName("Fraud Detection Analysis") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.memory.fraction", "0.9") \
        .getOrCreate()
    return spark

def main():
    spark = get_spark_session()
    train_df = load_data_from_cassandra(["invoice_train", "client_train"], spark)
    train_df = preprocess_data(train_df)
    # train_df = feature_engineering(train_df)

    model = build_and_evaluate_model(train_df)
    predictions = model.transform(train_df)

    # Save the predictions to a CSV file
    predictions.select("client_id", "invoice_date", "consumption_difference", "prediction").write.csv("/Users/csuftitan/Desktop/Coding/Big Data/FraudDetection/Outputs/predictions.csv", header=True, mode='overwrite')

    evaluate_model(predictions)

    numeric_cols = ['district', 'region', 'creation_year',
                    'tarif_type', 'target',
                    'reading_remarque', 'client_catg',
                    'old_index', 'new_index', 'consummation_level_1', 'consummation_level_3', 'counter_code']
    plot_correlation_matrix(train_df, numeric_cols)
    target_variable(train_df)

    # Example of hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(model.stages[-1].maxDepth, [5, 10, 20]) \
        .addGrid(model.stages[-1].numTrees, [20, 50, 100]) \
        .build()

    best_model = tune_model(train_df, model, param_grid)
    print("Best model obtained through tuning:", best_model)

if __name__ == "__main__":
    main()

