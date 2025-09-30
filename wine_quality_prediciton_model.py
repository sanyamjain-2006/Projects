import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('winequality-red.csv')

df.shape


df.isnull().sum()

df.info()

df
df.columns

df.describe()

sns.catplot(x='quality',data=df,kind='count ')

plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=df)

#volatile acidity=1/quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=df)

#quality is directly propotional to cirtic acid
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='residual sugar',data=df)

plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='pH',data=df)

corelation=df.corr(
    
)

plt.figure(figsize=(10,10))
sns.heatmap(corelation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')

x=df.drop("quality",axis=1)
y=df["quality"].apply(lambda y_value:1 if y_value>=7 else 0)

print(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=3)

print(x_train,y_train)

print(x_train.shape,
y_train.shape,
x_test.shape,
y_test.shape)

model=RandomForestClassifier()

model.fit(x_train,y_train)

x_test_prediction=model.predict(x_test)
test_data_accuracry=accuracy_score(x_test_prediction,y_test)
print("Accuracy of  Model is =",test_data_accuracry*100)

input_data=(8.1,0.38,0.28,2.1,0.066,13.0,30.0,0.9968,3.23,0.73,9.7)
input_data_arr=np.asarray(input_data)
input_data_reshaped=input_data_arr.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
if prediction[0]==1:
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")
