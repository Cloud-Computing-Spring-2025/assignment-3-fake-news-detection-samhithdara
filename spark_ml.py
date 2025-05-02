from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder \
    .appName("FakeNewsClassification") \
    .getOrCreate()

# -------------------- Task 1: Load & Basic Exploration --------------------
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")

# Show first 5 rows
df.show(5)

# Count total number of articles
print("Total Articles:", df.count())

# Retrieve distinct labels
df.select("label").distinct().show()

# Write DataFrame to CSV
df.write.csv("task1_output.csv", header=True, mode="overwrite")

# -------------------- Task 2: Text Preprocessing --------------------
# Convert text to lowercase
df_clean = df.withColumn("text", lower(df["text"]))

# Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized = tokenizer.transform(df_clean)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned = remover.transform(tokenized).select("id", "title", "filtered_words", "label")

# Convert array column to string for CSV
cleaned_for_csv = cleaned.withColumn("filtered_words_str", concat_ws(" ", cleaned["filtered_words"])) \
                         .select("id", "title", "filtered_words_str", "label")

cleaned_for_csv.write.csv("task2_output.csv", header=True, mode="overwrite")

# -------------------- Task 3: Feature Extraction --------------------
# Apply HashingTF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurized = hashingTF.transform(cleaned)

# Apply IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized)
rescaled = idf_model.transform(featurized)

# Index labels
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed = indexer.fit(rescaled).transform(rescaled)

# Select final columns
final_df = indexed.select("id", "filtered_words", "features", "label_index")

# For CSV, drop complex vector and array columns and just store id + label (optional)
final_df_for_csv = final_df.withColumn("filtered_words_str", concat_ws(" ", final_df["filtered_words"])) \
                           .select("id", "filtered_words_str", "label_index")

final_df_for_csv.write.csv("task3_output.csv", header=True, mode="overwrite")

# -------------------- Task 4: Model Training --------------------
# Split data
train, test = final_df.randomSplit([0.8, 0.2], seed=42)

# Train Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train)

# Generate predictions
predictions = model.transform(test)

# Write predictions to CSV
predictions_for_csv = predictions.select("id", "label_index", "prediction")
predictions_for_csv.write.csv("task4_output.csv", header=True, mode="overwrite")

# -------------------- Task 5: Evaluate the Model --------------------
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

# Print and save metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")

metrics_df = spark.createDataFrame([
    ("Accuracy", accuracy),
    ("F1 Score", f1_score)
], ["Metric", "Value"])

metrics_df.show()
metrics_df.write.csv("task5_output.csv", header=True, mode="overwrite")


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import lower
# from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.sql.functions import col

# # Initialize Spark
# spark = SparkSession.builder \
#     .appName("FakeNewsClassification") \
#     .getOrCreate()

# # -------------------- Task 1: Load & Basic Exploration --------------------
# df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
# df.createOrReplaceTempView("news_data")
# df.show(5)
# print("Total Articles:", df.count())
# df.select("label").distinct().show()
# df.write.csv("task1_output.csv", header=True)

# # -------------------- Task 2: Text Preprocessing --------------------
# df_clean = df.withColumn("text", lower(df["text"]))
# tokenizer = Tokenizer(inputCol="text", outputCol="words")
# tokenized = tokenizer.transform(df_clean)
# remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
# cleaned = remover.transform(tokenized).select("id", "title", "filtered_words", "label")
# from pyspark.sql.functions import concat_ws

# cleaned_string = cleaned.withColumn("filtered_words", concat_ws(" ", cleaned["filtered_words"]))
# cleaned_string.write.csv("task2_output.csv", header=True)


# # -------------------- Task 3: Feature Extraction --------------------
# hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
# tf = hashingTF.transform(cleaned)
# idf = IDF(inputCol="raw_features", outputCol="features")
# idf_model = idf.fit(tf)
# rescaled = idf_model.transform(tf)
# indexer = StringIndexer(inputCol="label", outputCol="label_index")
# indexed = indexer.fit(rescaled).transform(rescaled)
# final_df = indexed.select("id", "filtered_words", "features", "label_index")
# # Drop the features column before saving to CSV
# final_df_string = final_df.drop("features", "filtered_words")
# final_df_string.write.csv("task3_output.csv", header=True)



# # -------------------- Task 4: Model Training --------------------
# train, test = final_df.randomSplit([0.8, 0.2], seed=42)
# lr = LogisticRegression(featuresCol="features", labelCol="label_index")
# model = lr.fit(train)
# predictions = model.transform(test)
# predictions.select("id", "label_index", "prediction") \
#            .write.csv("task4_output.csv", header=True)


# # -------------------- Task 5: Evaluate the Model --------------------
# evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
# evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")
# accuracy = evaluator_acc.evaluate(predictions)
# f1_score = evaluator_f1.evaluate(predictions)

# metrics_df = spark.createDataFrame([
#     ("Accuracy", accuracy),
#     ("F1 Score", f1_score)
# ], ["Metric", "Value"])

# metrics_df.show()
# metrics_df.write.csv("task5_output.csv", header=True)
