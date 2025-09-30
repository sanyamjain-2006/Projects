# 🍷 Wine Quality Prediction (ML Project)

This project predicts the quality of wine using a **Random Forest Classifier** trained on the Wine Quality dataset.

---

## 📊 Dataset
- **Name**: Wine Quality Dataset (Red Wine)  
- **Source**: UCI Machine Learning Repository / Kaggle  
- File used: `winequality-red.csv`  

---

## 🚀 Project Workflow
1. **Data Exploration**  
   - Checked missing values, data types, shape, and statistics  
   - Visualized data with Seaborn & Matplotlib  
   - Correlation heatmap to understand feature relationships  

2. **Feature Engineering**  
   - Target column `quality` converted into binary classes:  
     - `1` → Good Quality (≥ 7)  
     - `0` → Bad Quality (< 7)  

3. **Model Training**  
   - Split data using `train_test_split`  
   - Trained a **Random Forest Classifier**  
   - Evaluated using **accuracy_score**

4. **Prediction**  
   - Model can predict wine quality for new input data  

---

## 📈 Results
- Achieved accuracy: ~ **(fill the accuracy you got in %)**  

---

## 🛠️ Tech Stack
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📂 Project Structure
