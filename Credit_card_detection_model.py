import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("creditcard.csv")
df

df.shape
df.isnull().sum()

df.shape


df.info()

df.describe()

pd.value_counts(df["Class"]==1)

in class columns 
0=Normal Transcation
1=Scam transcation


df["Class"].describe()

real=df[df.Class==0]
fake=df[df.Class==1]

print(real.shape,fake.shape)

plt.plot(real.Time,real.Amount,color="blue")
plt.plot(fake.Time,fake.Amount,color="red")

plt.show()

df["Amount"].describe()

real_Data=real.sample(n=492)

new_df=pd.concat([real_Data,fake],axis=0)

new_df.shape

new_df.isnull().sum()

pd.value_counts(new_df["Class"])

x=new_df.drop(["Class"],axis=1)
x
y=new_df["Class"]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

model=LogisticRegression()
model.fit(xtrain,ytrain)

prediction=model.predict(xtest)

accuracy=accuracy_score(ytest,prediction)*100
print(accuracy)
