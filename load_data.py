from pyspark.sql import SparkSession
from config import cassandra_config

def get_spark_session():
    spark = SparkSession.builder \
        .appName("Fraud Detection Analysis") \
        .config("spark.cassandra.connection.host", cassandra_config['host']) \
        .config("spark.cassandra.connection.port", cassandra_config['port']) \
        .config("spark.jars", "/Users/csuftitan/Desktop/Coding/Big Data/spark-cassandra-connector-master") \
        .getOrCreate()
    return spark


def load_data_from_cassandra(table_name, spark):
    df = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table=table_name, keyspace=cassandra_config['frauddetection']) \
        .load()
    return df