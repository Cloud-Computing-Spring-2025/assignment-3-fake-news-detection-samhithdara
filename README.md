# Assignment-5-FakeNews-Detection

This project builds a **fake news detection pipeline** using PySpark, including data preprocessing, feature extraction, model training, and evaluation.

---

### 🔧 Setup

1️.  **Install dependencies**
```bash
pip install pyspark
```

*(Or use a `requirements.txt` with `pyspark` inside.)*

2️.  **Prepare your virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

3️.  **Place your input file**
Ensure `fake_news_sample.csv` is in the project directory or use Dataset_Generator for generating data set.

---

### Tasks Breakdown

---

### **Task 1: Load & Basic Exploration**

- **Action:**
  - Load CSV into a Spark DataFrame.
  - Explore the first 5 rows.
  - Count the total number of articles.
  - Show distinct labels.
- **Output:**
  - Writes raw DataFrame to `task1_output.csv`.

---

### **Task 2: Text Preprocessing**

- **Action:**
  - Convert article text to lowercase.
  - Tokenize text into words.
  - Remove stopwords.
  - Create a new array column `filtered_words`.
  - Flatten `filtered_words` into `filtered_words_str` (space-separated string).
- **Output:**
  - Writes cleaned data to `task2_output.csv`.

---

### **Task 3: Feature Extraction**

- **Action:**
  - Use **HashingTF** to convert word arrays to raw feature vectors.
  - Apply **IDF** to scale features.
  - Encode labels with `StringIndexer`.
  - Select only necessary columns.
- **Output:**
  - Writes compacted dataset (without complex vector fields) to `task3_output.csv`.

---

### **Task 4: Model Training**

- **Action:**
  - Split dataset into training (80%) and testing (20%).
  - Train **Logistic Regression** using `features` and `label_index`.
  - Generate predictions on test data.
- **Output:**
  - Writes predictions (`id`, `label_index`, `prediction`) to `task4_output.csv`.

---

### **Task 5: Model Evaluation**

- **Action:**
  - Evaluate predictions using **accuracy** and **F1 score** with `MulticlassClassificationEvaluator`.
  - Display metrics.
- **Output:**
  - Writes evaluation metrics to `task5_output.csv`.

---

### 📂 Output Files

| Task | Output CSV                        |
|------|-----------------------------------|
| 1    | `task1_output.csv` (raw data)     |
| 2    | `task2_output.csv` (cleaned text) |
| 3    | `task3_output.csv` (features + label) |
| 4    | `task4_output.csv` (predictions)  |
| 5    | `task5_output.csv` (accuracy, F1) |

---

### 📦 Running the Pipeline

To run everything:
```bash
spark-submit spark_ml.py
```

---

###  Notes

- Spark cannot save arrays or vectors directly to CSV — you **must** flatten them into strings.  
- Always check `.columns` before selecting to avoid `UNRESOLVED_COLUMN` errors.  
- Use `mode="overwrite"` in `.write.csv()` to avoid folder existence errors.
