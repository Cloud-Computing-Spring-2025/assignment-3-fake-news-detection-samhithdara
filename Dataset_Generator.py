from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, rand, when, lit, expr, array_join, concat_ws  # Added concat_ws here
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()

# Task 1: Load & Basic Exploration
# Read the CSV file
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Register as temp view
df.createOrReplaceTempView("news_data")

# Show first 5 rows
print("First 5 rows:")
df.show(5)

# Count total articles
total_count = df.count()
print(f"Total number of articles: {total_count}")

# Get distinct labels
distinct_labels = df.select("label").distinct().collect()
print("Distinct labels:")
for label in distinct_labels:
    print(label["label"])

# Save Task 1 output
df.write.csv("task1_output.csv", header=True, mode="overwrite")

# Task 2: Text Preprocessing
# Use both title and text, but with limits
df_lower = df.withColumn("combined_text", 
                         concat_ws(" ", col("title"), col("text")))
df_lower = df_lower.withColumn("text_lower", lower(col("combined_text")))

# Tokenize title only
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
df_words = tokenizer.transform(df_lower)

# Limit to first 10 tokens to prevent overfitting on full text
# UDF to take only first N tokens
take_first_n = udf(lambda x: x[:10] if x and len(x) > 10 else x, ArrayType(StringType()))
df_words = df_words.withColumn("limited_words", take_first_n("words"))

# Remove stopwords
remover = StopWordsRemover(inputCol="limited_words", outputCol="filtered_words")
df_filtered = remover.transform(df_words)

# Select relevant columns
df_task2 = df_filtered.select("id", "title", "filtered_words", "label")

# Create temporary view (optional)
df_task2.createOrReplaceTempView("cleaned_news")

# Convert filtered_words array to string representation for CSV output
df_task2_output = df_filtered.select(
    "id", 
    "title", 
    array_join("filtered_words", ", ").alias("filtered_words"),
    "label"
)

# Save Task 2 output
df_task2_output.write.csv("task2_output.csv", header=True, mode="overwrite")

# Task 3: Feature Extraction 
# Use a moderate number of features
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=200)
df_tf = hashingTF.transform(df_filtered)

# IDF (Inverse Document Frequency)
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(df_tf)
df_tfidf = idfModel.transform(df_tf)

# Label indexing
labelIndexer = StringIndexer(inputCol="label", outputCol="label_index")
df_indexed = labelIndexer.fit(df_tfidf).transform(df_tfidf)

# Select relevant columns for output
df_task3_output = df_indexed.select(
    "id", 
    array_join("filtered_words", ", ").alias("filtered_words"), 
    col("features").cast("string").alias("features"), 
    "label_index"
)

# Save Task 3 output
df_task3_output.write.csv("task3_output.csv", header=True, mode="overwrite")

# Task 4: Model Training
# Remove the noisy label manipulation and use original labels
train_data, test_data = df_indexed.randomSplit([0.7, 0.3], seed=42)

# Print counts to verify split
print(f"Training set size: {train_data.count()}")
print(f"Test set size: {test_data.count()}")

# Use moderate regularization parameters
lr = LogisticRegression(
    featuresCol="features", 
    labelCol="label_index",  # Use original labels
    maxIter=5,  # Moderate number of iterations
    regParam=0.3,  # Moderate regularization
    elasticNetParam=0.5  # Balanced L1/L2 mix
)

# Train model
lr_model = lr.fit(train_data)

# Make predictions on test data with original labels
predictions = lr_model.transform(test_data)

# Select relevant columns for output
df_task4 = predictions.select("id", "title", "label_index", "prediction")

# Save Task 4 output
df_task4.write.csv("task4_output.csv", header=True, mode="overwrite")

# Task 5: Evaluate the Model
# Evaluate against original labels (not noisy ones)
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label_index",  # Use original labels for evaluation 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator_acc.evaluate(predictions)

# Calculate F1 score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_index", 
    predictionCol="prediction", 
    metricName="f1"
)
f1_score = evaluator_f1.evaluate(predictions)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")

# Create metrics dataframe
metrics_data = [
    ("Accuracy", accuracy),
    ("F1 Score", f1_score)
]

schema = StructType([
    StructField("Metric", StringType(), True),
    StructField("Value", DoubleType(), True)
])

metrics_df = spark.createDataFrame(metrics_data, schema)

# Save Task 5 output
metrics_df.write.csv("task5_output.csv", header=True, mode="overwrite")

# Stop Spark session
spark.stop()