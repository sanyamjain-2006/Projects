# 💳 Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent credit card transactions using Machine Learning.  
Since financial fraud datasets are highly imbalanced (most transactions are legitimate), this model focuses on improving fraud detection accuracy using **data resampling** and **advanced evaluation metrics** like Precision, Recall, F1-score, and ROC-AUC.

---

## 🚀 Project Overview
Credit card fraud is a major problem for financial institutions and users.  
This project uses historical transaction data to train a model that can predict whether a given transaction is **Legitimate (0)** or **Fraudulent (1)**.

---

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**
   - Loaded dataset using pandas  
   - Checked for null values and data distribution  
   - Scaled features using `StandardScaler` (since values vary widely)  

2. **Handling Class Imbalance**
   - Used `SMOTE` (Synthetic Minority Oversampling Technique) to balance classes  
   - Verified balanced distribution after resampling  

3. **Model Training**
   - Models tested:
     - Logistic Regression  
     - Random Forest  
     - Decision Tree  
     - XGBoost (optional)
   - Used `train_test_split` with `stratify=y` to maintain class ratio  

4. **Model Evaluation**
   - Metrics used:
     - Precision  
     - Recall  
     - F1-score  
     - ROC-AUC Score  
   - Accuracy was not the main focus due to class imbalance.

---

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807  
- **Fraudulent Transactions:** 492  
- **Legitimate Transactions:** 284,315  
- **Features:** 30 columns (V1–V28 are PCA components + Amount + Time)  

---

## ⚙️ Tech Stack
- **Language:** Python 🐍  
- **Libraries Used:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`
  

---


