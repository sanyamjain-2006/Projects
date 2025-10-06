This project is a Machine Learning model to predict the presence of heart disease based on patient data.
It uses Random Forest Classifier to classify whether a person has heart disease (target = 1) or not (target = 0).

📂 Dataset

The dataset contains medical information of patients such as:

Age, Sex, Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Cholesterol level (chol)

Fasting Blood Sugar (fbs)

Resting ECG results (restecg)

Maximum heart rate achieved (thalach)

Slope of ST segment (slope)

Thalassemia (thal)

Target variable: target (1 = Heart Disease, 0 = No Heart Disease)

Source: UCI Heart Disease Dataset

🛠️ Technologies Used

Python 3

Libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

🧩 Features

Data exploration (df.head(), df.info(), df.describe())

Missing value check and basic cleaning

Data visualization:

Line plots

Heatmaps for correlation

Train/Test split using train_test_split

Random Forest Classifier for prediction

Model evaluation using accuracy score

⚡ How to Run

Clone the repository or download the files.

Install required libraries:

pip install pandas numpy scikit-learn matplotlib seaborn


Place heart_disease_data.csv in the same directory as the notebook.

Run the Python script or Jupyter Notebook:

python heart_disease_prediction.py


or open heart_disease_prediction.ipynb in Jupyter Notebook / Google Colab.

📊 Model Performance

The trained Random Forest model predicts whether a patient has heart disease based on their medical data.

Accuracy can vary depending on dataset and train/test split.

🔮 Future Improvements

Hyperparameter tuning for Random Forest

Try other classifiers (Logistic Regression, XGBoost, SVM)

Evaluate using precision, recall, F1-score

Feature importance visualization

Save the trained model for deployment
