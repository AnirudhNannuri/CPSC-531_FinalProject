from pyspark.pandas import DataFrame
from pyspark.sql.functions import year, col
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_consumption_levels(df):
    # Calculate total consumption if not already present
    df = df.withColumn("total_consumption", col("consommation_level_1") + col("consommation_level_2") +
                       col("consommation_level_3") + col("consommation_level_4"))

    pd_df = df.select("total_consumption").toPandas()
    plt.figure(figsize=(10, 6))
    pd_df['total_consumption'].hist(bins=50)
    plt.title('Distribution of Total Consumption')
    plt.xlabel('Total Consumption')
    plt.ylabel('Frequency')
    plt.show()

def target_variable(df):
    # Assuming 'invoice_date' has been converted to a datetime type
    df = df.withColumn("year", year("invoice_date"))
    client_counts = df.groupBy("year").count().orderBy("year")
    pd_df = client_counts.toPandas()

    plt.figure(figsize=(10, 6))
    plt.plot(pd_df['clients'], pd_df['target'], marker='o')
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target Variable')
    plt.ylabel('Number of Clients')
    plt.grid(True)
    plt.show()

def plot_proportion_of_counter_types(df):
    counter_types = df.groupBy("counter_type").count()
    pd_df = counter_types.toPandas()

    plt.figure(figsize=(10, 6))
    plt.pie(pd_df['count'], labels=pd_df['counter_type'], autopct='%1.1f%%')
    plt.title('Proportion of Counter Types')
    plt.show()

def plot_correlation_matrix(df: DataFrame, numeric_cols):
    # Create an empty list to hold correlation values
    correlation_data = []

    # Compute the correlation for each pair of numeric columns
    for col1 in numeric_cols:
        temp = []
        for col2 in numeric_cols:
            corr_value = df.stat.corr(col1, col2)  # Compute correlation
            temp.append(corr_value)
        correlation_data.append(temp)

    # Convert correlation data to a Pandas DataFrame
    corr_df = pd.DataFrame(correlation_data, index=numeric_cols, columns=numeric_cols)

    # Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title("Correlations in Train Dataset (Numeric Columns)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()