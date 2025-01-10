from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when, to_date, date_format, monotonically_increasing_id, isnan
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression

# Initialize Spark session
spark = SparkSession.builder.appName("TurbinePipeline").getOrCreate()

# ----------------------- Pipeline Functions -----------------------

def load_bronze_data(input_path):
    """Load raw data into the Bronze layer."""
    bronze_schema = StructType([
        StructField("timestamp", StringType(), True),
        StructField("turbine_id", IntegerType(), True),
        StructField("wind_speed", DoubleType(), True),
        StructField("wind_direction", DoubleType(), True),
        StructField("power_output", DoubleType(), True),
    ])
    return spark.read.format("csv").option("header", "true").schema(bronze_schema).load(input_path)

def validate_bronze_data(df):
    """Perform data quality checks on the Bronze layer."""
    # Check for duplicates
    if df.count() != df.dropDuplicates().count():
        print("Soft warning: Duplicate rows found in Bronze Layer!")

    # Check for nulls in critical fields
    critical_fields = ["timestamp", "turbine_id", "power_output"]
    null_critical_fields = (
        df.select([col(f).isNull().alias(f"{f}_is_null") for f in critical_fields])
        .filter(
            col(f"{critical_fields[0]}_is_null") |
            col(f"{critical_fields[1]}_is_null") |
            col(f"{critical_fields[2]}_is_null")
        )
        .count()
    )
    if null_critical_fields > 0:
        raise ValueError(f"Null values found in critical fields of Bronze Layer! ({null_critical_fields} rows)")

    # Check for invalid dates
    invalid_dates = (
        df.withColumn("date", to_date("timestamp"))
        .filter(col("date").isNull())
        .count()
    )
    if invalid_dates > 0:
        print(f"Invalid or null timestamps found in {invalid_dates} rows!")

    # Check for unreasonable ranges
    wind_speed_range = (0, 100)
    wind_direction_range = (0, 360)
    power_output_range = (0, 5000)

    out_of_range_count = df.filter(
        (col("wind_speed") < wind_speed_range[0]) |
        (col("wind_speed") > wind_speed_range[1]) |
        (col("wind_direction") < wind_direction_range[0]) |
        (col("wind_direction") > wind_direction_range[1]) |
        (col("power_output") < power_output_range[0]) |
        (col("power_output") > power_output_range[1])
    ).count()

    if out_of_range_count > 0:
        print(f"Soft warning: Found {out_of_range_count} rows with out-of-range values in Bronze Layer!")

    # Check for NaN values
    nan_count = (
        df.select(
            [isnan(col(f)).alias(f"{f}_is_nan") for f in ["wind_speed", "wind_direction", "power_output"]]
        )
        .filter(
            col("wind_speed_is_nan") |
            col("wind_direction_is_nan") |
            col("power_output_is_nan")
        )
        .count()
    )
    if nan_count > 0:
        print(f"Soft warning: NaN values found in numerical fields ({nan_count} rows)!")
    return df

def save_bronze_layer(df, output_table):
    """Save the Bronze layer data."""
    df.write.format("delta").mode("overwrite").partitionBy("turbine_id").saveAsTable(output_table)

def process_silver_layer(bronze_df):
    """Transform and clean data for the Silver layer."""
    bronze_df = bronze_df.dropDuplicates()
    turbine_window = Window.partitionBy("turbine_id")

    filled_df = (
        bronze_df
        .withColumn(
            "wind_speed",
            F.when(F.col("wind_speed").isNull(), F.avg("wind_speed").over(turbine_window))
            .otherwise(F.col("wind_speed")),
        )
        .withColumn(
            "wind_direction",
            F.when(F.col("wind_direction").isNull(), F.avg("wind_direction").over(turbine_window))
            .otherwise(F.col("wind_direction")),
        )
        .withColumn(
            "power_output",
            F.when(F.col("power_output").isNull(), F.avg("power_output").over(turbine_window))
            .otherwise(F.col("power_output")),
        )
    )

    stats_columns = ["wind_speed", "wind_direction", "power_output"]
    for column in stats_columns:
        q1, q3 = filled_df.approxQuantile(column, [0.25, 0.75], 0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filled_df = filled_df.filter(
            (col(column) >= lower_bound) & (col(column) <= upper_bound)
        )
    return filled_df

def save_silver_layer(df, output_table):
    """Save the Silver layer data."""
    df.write.format("delta").mode("overwrite").partitionBy("turbine_id").saveAsTable(output_table)

def process_gold_layer(silver_df):
    """Transform data for the Gold layer (fact and dimension tables)."""
    silver_df_with_keys = silver_df.withColumn(
        "record_sid", monotonically_increasing_id()
    ).withColumn("date_key", date_format(to_date("timestamp"), "yyyyMMdd"))

    fact_turbine_metrics = silver_df_with_keys.select(
        "record_sid", "turbine_id", "wind_speed", "wind_direction", "power_output", "timestamp", "date_key"
    )

    dim_time_df = (
        silver_df_with_keys.select(to_date("timestamp").alias("date"))
        .distinct()
        .withColumn("date_key", date_format("date", "yyyyMMdd"))
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("day", F.dayofmonth("date"))
        .withColumn("day_of_week", F.dayofweek("date"))
        .withColumn("quarter", F.quarter("date"))
        .select("date_key", "date", "year", "month", "day", "day_of_week", "quarter")
    )

    dim_turbine_df = (
        silver_df_with_keys.select("turbine_id")
        .distinct()
        .withColumn("location", lit("unknown"))
        .withColumn("manufacturer", lit("unknown"))
        .withColumn("capacity", lit("unknown"))
    )
    return fact_turbine_metrics, dim_time_df, dim_turbine_df

def save_gold_layer(fact_table, time_dim, turbine_dim, fact_table_name, time_dim_name, turbine_dim_name):
    """Save Gold layer tables."""
    fact_table.write.format("delta").mode("overwrite").saveAsTable(fact_table_name)
    time_dim.write.format("delta").mode("overwrite").saveAsTable(time_dim_name)
    turbine_dim.write.format("delta").mode("overwrite").saveAsTable(turbine_dim_name)

def run_pipeline():
    """Run the data pipeline (Bronze, Silver, Gold layers)."""
    bronze_path = "/FileStore/colibri/*.csv"
    bronze_table = "bronze_turbine_data"
    silver_table = "silver_turbine_data"
    fact_table_name = "gold_fact_turbine_metrics"
    time_dim_name = "gold_dim_time"
    turbine_dim_name = "gold_dim_turbine"

    # Step 1: Bronze Layer
    bronze_df = load_bronze_data(bronze_path)
    bronze_df = validate_bronze_data(bronze_df)
    save_bronze_layer(bronze_df, bronze_table)

    # Step 2: Silver Layer
    silver_df = process_silver_layer(bronze_df)
    save_silver_layer(silver_df, silver_table)

    # Step 3: Gold Layer
    fact_table, time_dim, turbine_dim = process_gold_layer(silver_df)
    save_gold_layer(fact_table, time_dim, turbine_dim, fact_table_name, time_dim_name, turbine_dim_name)

# ----------------------- Stats Functions -----------------------

def calculate_summary_stats():
    """Calculate summary statistics and save results."""
    fact_table_name = "gold_fact_turbine_metrics"
    summary_table = "gold_daily_summary"

    fact_table = spark.table(fact_table_name)
    daily_summary_df = fact_table.groupBy("turbine_id", "date_key").agg(
        F.min("power_output").alias("min_power_output"),
        F.max("power_output").alias("max_power_output"),
        F.avg("power_output").alias("avg_power_output"),
    )
    daily_summary_df.write.format("delta").mode("overwrite").saveAsTable(summary_table)

def detect_anomalies():
    """Detect anomalies in the fact table."""
    fact_table_name = "gold_fact_turbine_metrics"
    anomalies_table = "gold_anomalies"

    fact_table = spark.table(fact_table_name)
    stats_df = fact_table.groupBy("turbine_id", "date_key").agg(
        F.mean("power_output").alias("mean_power_output"),
        F.stddev("power_output").alias("stddev_power_output"),
    )
    joined_df = fact_table.join(stats_df, on=["turbine_id", "date_key"], how="inner")
    anomalies_df = joined_df.filter(
        (col("power_output") > col("mean_power_output") + 2 * col("stddev_power_output")) |
        (col("power_output") < col("mean_power_output") - 2 * col("stddev_power_output"))
    ).select(
        "turbine_id", "date_key", "power_output", "mean_power_output", "stddev_power_output"
    )
    anomalies_df.write.format("delta").mode("overwrite").saveAsTable(anomalies_table)

# ----------------------- ML Functions -----------------------

def train_regression_model():
    """Train a Linear Regression model."""
    fact_table_name = "gold_fact_turbine_metrics"
    regression_model_path = "/models/linear_regression"
    features = ["wind_speed", "wind_direction"]
    label = "power_output"

    fact_table = spark.table(fact_table_name)
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data = assembler.transform(fact_table).select("features", label)
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol=label)
    lr_model = lr.fit(train_data)
    lr_model.write().overwrite().save(regression_model_path)
    print("Linear Regression model saved at:", regression_model_path)

def train_classification_model():
    """Train a Logistic Regression model."""
    fact_table_name = "gold_fact_turbine_metrics"
    classification_model_path = "/models/logistic_regression"
    features = ["wind_speed", "wind_direction"]
    label = "power_output"
    threshold = 100

    fact_table = spark.table(fact_table_name)
    labeled_df = fact_table.withColumn(
        "label", when(col(label) < threshold, 1).otherwise(0)
    )
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data = assembler.transform(labeled_df).select("features", "label")
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_data)
    lr_model.write().overwrite().save(classification_model_path)
    print("Logistic Regression model saved at:", classification_model_path)

# ----------------------- Main Functions -----------------------

if __name__ == "__main__":
    # Choose which task to run
    task = input("Enter task (pipeline/stats/ml): ").strip().lower()

    if task == "pipeline":
        run_pipeline()
    elif task == "stats":
        calculate_summary_stats()
        detect_anomalies()
    elif task == "ml":
        train_regression_model()
        train_classification_model()
    else:
        print("Invalid task! Choose 'pipeline', 'stats', or 'ml'.")
