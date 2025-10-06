import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



df=pd.read_csv('heart_disease_data.csv')
df.head()

df.info()

df.isnull().sum()

plt.figure(figsize=(10,10))
plt.plot(df["cp"],df["target"])
plt.xlabel("cp")
plt.ylabel("target")
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)

df.shape

df.drop_duplicates()

df.columns

df["sex"].value_counts()

df.drop(columns=["exang","oldpeak","ca"],axis=1,inplace=True)

df.shape

df["target"].value_counts()

x=df.drop("target",axis=1)
y=df["target"]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()

model.fit(xtrain,ytrain)

prediciton=model.predict(xtest)

print(prediciton)

accuracy=accuracy_score(ytest,prediciton)*100

print(f"the prediciton of model is {accuracy}")
