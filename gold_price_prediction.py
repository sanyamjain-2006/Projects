import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("gld_price_data.csv")
df

df.shape

df.isnull().sum()

df.info()
df.describe()

x=df["Date"]
y=df["GLD"]
x,y

plt.figure(figsize=(10,5))
plt.plot(x,y)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Gold Price")
plt.show()

plt.plot(df["GLD"],df["SLV"])
plt.xlabel("Gold Price")
plt.ylabel("Silver Price")
plt.title("Gold vs Silver Price")
plt.show()

x_=df.drop(columns=["Date","GLD"],axis=1)
y_=df["GLD"]
x_,y_

plt.plot(x_,y_)
plt.show()

xtrain,xtest,ytrain,ytest=train_test_split(x_,y_,test_size=0.2,random_state=2)
print(xtrain,ytrain)

model=RandomForestRegressor()

modleing=model.fit(xtrain,ytrain)
prediction=model.predict(xtest)
predicition

print(len(ytest),len(prediction))
perce=metrics.r2_score(ytest,prediction)
print(perce*100)

plt.scatter(ytest,prediction)
plt.xlabel("Gold Price")
plt.ylabel("Predicted Gold Price")
plt.title("Gold Price Prediction")
plt.xlim(100,120)
plt.ylim(100,120)
plt.show()
