# Databricks notebook source
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col,
    lit,
    when,
    year,
    month,
    dayofmonth,
    hour,
    dayofweek,
    quarter,
    min,
    max,
    avg,
    stddev,
    variance,
    count,
    mean,
    to_date,
    date_format,
    monotonically_increasing_id,
    isnan,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Bronze Layer - Ingestion

# COMMAND ----------

# Define schema for Bronze Layer
bronze_schema = StructType(
    [
        StructField("timestamp", StringType(), True),
        StructField("turbine_id", IntegerType(), True),
        StructField("wind_speed", DoubleType(), True),
        StructField("wind_direction", DoubleType(), True),
        StructField("power_output", DoubleType(), True),
    ]
)

# Step 1: Bronze Layer
bronze_df = (
    spark.read.format("csv")
    .option("header", "true")
    .schema(bronze_schema)
    .load("/FileStore/colibri/*.csv")
)

# Data Quality Checks

# 1. Check for duplicate rows
if bronze_df.count() != bronze_df.dropDuplicates().count():
    print("Soft warning: Duplicate rows found in Bronze Layer!")

# 2. Check for nulls in critical fields
critical_fields = ["timestamp", "turbine_id", "power_output"]
null_critical_fields = (
    bronze_df.select([col(f).isNull().alias(f"{f}_is_null") for f in critical_fields])
    .filter(
        col(f"{critical_fields[0]}_is_null")
        | col(f"{critical_fields[1]}_is_null")
        | col(f"{critical_fields[2]}_is_null")
    )
    .count()
)

if null_critical_fields > 0:
    raise ValueError(
        f"Null values found in critical fields of Bronze Layer! ({null_critical_fields} rows)"
    )

# 3. Check for invalid or null dates
invalid_dates = (
    bronze_df.withColumn("date", to_date("timestamp"))
    .filter(col("date").isNull())
    .count()
)
if invalid_dates > 0:
    print(f"Invalid or null timestamps found in {invalid_dates} rows!")

# 4. Check for unreasonable ranges in numerical fields
wind_speed_range = (0, 100)  # Example: 0 to 100 m/s
wind_direction_range = (0, 360)  # Example: 0 to 360 degrees
power_output_range = (0, 5000)  # Example: 0 to 5000 MW

out_of_range_count = bronze_df.filter(
    (col("wind_speed") < wind_speed_range[0])
    | (col("wind_speed") > wind_speed_range[1])
    | (col("wind_direction") < wind_direction_range[0])
    | (col("wind_direction") > wind_direction_range[1])
    | (col("power_output") < power_output_range[0])
    | (col("power_output") > power_output_range[1])
).count()

if out_of_range_count > 0:
    print(
        f"Soft warning: Found {out_of_range_count} rows with out-of-range values in Bronze Layer!"
    )

# 5. Check for non-numeric or NaN values in numerical fields
nan_count = (
    bronze_df.select(
        [
            isnan(col(f)).alias(f"{f}_is_nan")
            for f in ["wind_speed", "wind_direction", "power_output"]
        ]
    )
    .filter(
        col("wind_speed_is_nan")
        | col("wind_direction_is_nan")
        | col("power_output_is_nan")
    )
    .count()
)

if nan_count > 0:
    print(f"Soft warning: NaN values found in numerical fields ({nan_count} rows)!")

# Save Bronze Layer
bronze_df.write.format("delta").mode("overwrite").partitionBy("turbine_id").saveAsTable(
    "bronze_turbine_data"
)
# bronze_df.display()


# COMMAND ----------

# MAGIC %md
# MAGIC Silver layer - Processing

# COMMAND ----------

# Step 3: Silver

bronze_df = bronze_df.dropDuplicates()

# Calculate average values within each turbine_id group
turbine_window = Window.partitionBy("turbine_id")
filled_df = (
    bronze_df.withColumn(
        "wind_speed",
        F.when(
            F.col("wind_speed").isNull(), F.avg("wind_speed").over(turbine_window)
        ).otherwise(F.col("wind_speed")),
    )
    .withColumn(
        "wind_direction",
        F.when(
            F.col("wind_direction").isNull(),
            F.avg("wind_direction").over(turbine_window),
        ).otherwise(F.col("wind_direction")),
    )
    .withColumn(
        "power_output",
        F.when(
            F.col("power_output").isNull(), F.avg("power_output").over(turbine_window)
        ).otherwise(F.col("power_output")),
    )
)

### ANOMALY REMOVAL ###

# Define the columns to process
stats_columns = ["wind_speed", "wind_direction", "power_output"]

# Initialize cleaned DataFrame with the original filled DataFrame
cleaned_df = filled_df

for column in stats_columns:
    # Calculate Q1, Q3, and IQR for the column
    q1, q3 = filled_df.approxQuantile(column, [0.25, 0.75], 0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter rows based on the IQR bounds
    silver_df = cleaned_df.filter(
        (col(column) >= lower_bound) & (col(column) <= upper_bound)
    )

# Save Silver Layer
silver_df.write.format("delta").mode("overwrite").partitionBy("turbine_id").saveAsTable(
    "silver_turbine_data"
)
# silver_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Gold layer - Processing & Warehousing

# COMMAND ----------


# Step 3: Gold Layer (Star Schema)

# Create a fact table
silver_df_with_keys = silver_df.withColumn(
    "record_sid", monotonically_increasing_id()
).withColumn("date_key", date_format(to_date("timestamp"), "yyyyMMdd"))

fact_turbine_metrics = silver_df_with_keys.select(
    "record_sid",
    "turbine_id",
    "wind_speed",
    "wind_direction",
    "power_output",
    "timestamp",
    "date_key",
)

# Dimension: Time
dim_time_df = (
    cleaned_df.select(to_date("timestamp").alias("date"))
    .distinct()
    .withColumn("date_key", date_format("date", "yyyyMMdd"))
    .withColumn("year", year("date"))
    .withColumn("month", month("date"))
    .withColumn("day", dayofmonth("date"))
    .withColumn("day_of_week", dayofweek("date"))
    .withColumn("quarter", quarter("date"))
    .select("date_key", "date", "year", "month", "day", "day_of_week", "quarter")
)

# Dimension: Turbine
dim_turbine_df = (
    cleaned_df.select("turbine_id")
    .distinct()
    .withColumn(
        "location", lit("unknown")
    )  # Replace "unknown" with actual defaults if available
    .withColumn(
        "manufacturer", lit("unknown")
    )  # Replace "unknown" with actual defaults if available
    .withColumn(
        "capacity", lit("unknown")
    )  # Replace "unknown" with actual defaults if available
)

# Save the fact and dimension tables
fact_turbine_metrics.write.format("delta").mode("overwrite").saveAsTable(
    "gold_fact_turbine_metrics"
)
dim_time_df.write.format("delta").mode("overwrite").saveAsTable("gold_dim_time")
dim_turbine_df.write.format("delta").mode("overwrite").saveAsTable("gold_dim_turbine")

# fact_turbine_metrics.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Summary statistics

# COMMAND ----------

# Define the time period (e.g., extract the date for grouping by 24 hours)
daily_summary_df = fact_turbine_metrics.groupBy("turbine_id", "date_key").agg(
    min("power_output").alias("min_power_output"),
    max("power_output").alias("max_power_output"),
    avg("power_output").alias("avg_power_output"),
)

daily_summary_df.write.format("delta").mode("overwrite").saveAsTable(
    "gold_daily_summary"
)
# daily_summary_df.display()

# COMMAND ----------


# Step 1: Calculate mean and standard deviation for each turbine over a given period (e.g., daily)
stats_df = fact_turbine_metrics.groupBy("turbine_id", "date_key").agg(
    mean("power_output").alias("mean_power_output"),
    stddev("power_output").alias("stddev_power_output"),
)

# Step 2: Join the calculated statistics back to the fact table
joined_df = fact_turbine_metrics.join(
    stats_df, on=["turbine_id", "date_key"], how="inner"
)

# Step 3: Identify anomalies (output outside ±2 standard deviations from the mean)
anomalies_df = joined_df.filter(
    (col("power_output") > col("mean_power_output") + 2 * col("stddev_power_output"))
    | (col("power_output") < col("mean_power_output") - 2 * col("stddev_power_output"))
).select(
    "turbine_id", "date_key", "power_output", "mean_power_output", "stddev_power_output"
)

# Show the anomalies
# anomalies_df.display()
anomalies_df.write.format("delta").mode("overwrite").saveAsTable("gold_anomalies")


# COMMAND ----------

# MAGIC %md
# MAGIC Machine learning

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Prepare the data
features = ["wind_speed", "wind_direction"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(fact_turbine_metrics).select("features", "power_output")

# Split into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="power_output")
lr_model = lr.fit(train_data)

# Evaluate the model
predictions = lr_model.transform(test_data)
# predictions.select("features", "power_output", "prediction").display()


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Label data for anomalies (e.g., power_output < threshold)
threshold = 100  # Example threshold
labeled_df = fact_turbine_metrics.withColumn(
    "label", when(col("power_output") < threshold, 1).otherwise(0)
)

# Prepare the data
features = ["wind_speed", "wind_direction"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(labeled_df).select("features", "label")

# Split into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train a Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Evaluate the model
predictions = lr_model.transform(test_data)
# predictions.select("features", "label", "prediction", "probability").display()
