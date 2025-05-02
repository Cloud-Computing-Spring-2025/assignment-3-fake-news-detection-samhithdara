# Assignment-5-FakeNews-Detection

This project implements a fake news detection pipeline using **Apache Spark**. It processes a sample dataset, performs text preprocessing, extracts features, trains a logistic regression model, and evaluates its performance.  

---

## 🗂 **Project Structure**
- **Input file:** `fake_news_sample.csv`
- **Outputs:**
  - `task1_output.csv` → Raw data exploration results
  - `task2_output.csv` → Cleaned + preprocessed text
  - `task3_output.csv` → Extracted features + indexed labels
  - `task4_output.csv` → Model predictions on test data
  - `task5_output.csv` → Evaluation metrics (accuracy, F1 score)

---

## 🔧 **Pipeline Overview**

### **Task 1: Load & Explore Data**
- Load CSV into Spark DataFrame.
- Show first 5 rows.
- Count total number of articles.
- List distinct labels.
- Save raw data snapshot to `task1_output.csv`.

---

### **Task 2: Text Preprocessing**
- Combine `title` + `text` columns into a unified field.
- Convert combined text to lowercase.
- Tokenize the text.
- Limit to **first 10 tokens** to prevent overfitting on long articles.
- Remove stopwords.
- Save preprocessed output to `task2_output.csv`.

---

### **Task 3: Feature Extraction**
- Apply `HashingTF` (with 200 features).
- Apply `IDF` (inverse document frequency).
- Index string labels to numerical indices.
- Save final features and labels to `task3_output.csv`.

---

### **Task 4: Model Training**
- Split dataset into **70% train / 30% test**.
- Train a logistic regression model with:
  - `maxIter = 5`
  - `regParam = 0.3`
  - `elasticNetParam = 0.5` (mix of L1/L2 regularization).
- Make predictions on test set.
- Save predictions to `task4_output.csv`.

---

### **Task 5: Model Evaluation**
- Compute **Accuracy**.
- Compute **F1 Score**.
- Save evaluation metrics to `task5_output.csv`.

---

##  **How to Run**
1. Ensure you have:
   - Python 3.x
   - Apache Spark + PySpark installed
2. Place the `fake_news_sample.csv` file in the same directory.
3. Run:
   ```bash
   spark-submit fake_news_detection.py
   ```
4. Check the generated output CSV files.

---

## 📁 **Outputs Explanation**
| File                  | Description                                               |
|-----------------------|-----------------------------------------------------------|
| `task1_output.csv`    | Raw data with all columns from the original CSV.           |
| `task2_output.csv`    | Preprocessed and cleaned text after tokenizing + stopword removal. |
| `task3_output.csv`    | Feature vectors and numerical labels ready for training.  |
| `task4_output.csv`    | Model predictions on test set (includes predicted labels).|
| `task5_output.csv`    | Overall accuracy and F1 score of the trained model.       |

---

## 📊 **Model Details**
- **Model type:** Logistic Regression (multiclass)
- **Features:** TF-IDF vectors from top 200 hashed terms (after preprocessing)
- **Evaluation:** Accuracy and F1 score

