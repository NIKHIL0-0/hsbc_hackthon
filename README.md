# Fraud Detection Model

This project implements a **fraud detection pipeline** that processes transaction data, engineers advanced features, trains and evaluates multiple machine learning models, and predicts fraud on new datasets.

---

## Overview

The pipeline:
1. Loads and examines transaction data.
2. Performs feature engineering (customer, merchant, category, and time-based features).
3. Prepares features with preprocessing (scaling, encoding, imputation).
4. Trains and evaluates multiple ML models (Logistic Regression, Random Forest, Gradient Boosting).
5. Selects the best model (Random Forest in this case).
6. Saves the trained model and preprocessing pipeline.
7. Predicts fraud labels and probabilities on new datasets.

---

## Project Structure

### **1. Data Preparation**
- **`load_and_examine_data`**  
  Loads the dataset, shows shape, types, missing values, and fraud distribution.
- **`advanced_feature_engineering`**  
  Creates new behavioral and risk features:
  - Customer & Merchant statistics (totals, averages, diversity).
  - Time features (hour, day, week).
  - Amount transformations (log, sqrt, bins).
  - Demographic and geo features (age grouping, same zip flags).
  - Ratios like `amount_vs_customer_mean` and risk scores.
- **`prepare_features`**  
  Prepares numeric and categorical columns, fills missing values, and returns `X` and `y`.

### **2. Preprocessing & Model Training**
- **`create_preprocessing_pipeline`**  
  Handles numeric scaling, imputation, and one-hot encoding for categoricals.
- **`train_models`**  
  Trains Logistic Regression, Random Forest, and Gradient Boosting models, compares AUC, precision, recall, F1-score, and confusion matrices.

### **3. Model Evaluation & Feature Analysis**
- Selects the model with the best validation AUC (Random Forest).
- Evaluates with AUC, precision, recall, F1, confusion matrix.
- **`plot_feature_importance`** shows top features driving predictions.
- **`save_model_and_preprocessor`** saves the pipeline for future use.

### **4. Predicting on New Data**
- **`predict_on_test_data(test_path)`**  
  Loads test data, applies the same feature engineering, and generates:
  - Fraud labels (0/1),
  - Fraud probabilities,
  - Risk levels (Low, Medium, High).  
  Outputs results to `test_predictions.csv`.

---

## Insights from the Data

### **1. Transaction Amount Patterns (Log-Scaled)**
- Non-fraudulent transactions cluster around **log(amount) ≈ 4**.
- Fraudulent transactions are **fewer but skew toward higher amounts**, suggesting frauds often occur at unusually large amounts.

### **2. Merchant Popularity & Fraud Frequency**
- Non-fraud dominates across all merchant popularity levels.
- Fraud is **more concentrated in less popular merchants** but present everywhere.
- Log-scale shows the long-tail nature (few merchants handle most transactions).

### **3. Age Group & Fraud Rates**
- Most transactions come from groups **‘2’ (168,425) and ‘3’ (132,505)**.
- Fraud rate is fairly balanced (1–1.2%) across groups.
- **Group ‘0’ has the highest fraud rate (1.86%)**, while unknown (‘U’) has the lowest (0.56%).

---

