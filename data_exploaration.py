import pandas as pd
import numpy as np

df=pd.read_csv('titanic_dataset.csv')
print(df.head())
print(df.info())
# Looking for Null values
print(df.isna().sum())

from sklearn.impute import SimpleImputer
s=SimpleImputer(strategy='mean')
df['Age']=s.fit_transform(df[['Age']])

# removing unwanted columns
df.drop(['Embarked','Cabin'],axis=1,inplace=True)
print(df.head())

import hashlib

tkt = df["Ticket"]
df.drop('Ticket',axis=1,inplace=True)
def hash_encode(value):
    return int(hashlib.md5(value.encode()).hexdigest(), 16) % (10**8)  # Keep it within a range

df["Ticket_Hash"] = tkt.apply(hash_encode)
print(df.head())

# combining sibsp and parch columns into single travelled alone column
df['family']=df['SibSp']+df['Parch']
df.loc[df['family']>0,'travelled Alone']=0
df.loc[df['family']==0,'travelled Alone']=1
df.drop(['SibSp','Parch'],axis=1)
print(df.head())

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse_output=False)
df['Sex'] = oh.fit_transform(df[['Sex']])
print(df.head())



import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Fare'], color='blue', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare Scatter Plot')
plt.draw()

plt.figure(figsize=(6, 4))
plt.hist(df['Survived'], bins=20, color='green', alpha=0.7)
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.title('density of passengers survived and not survived')
plt.draw()

plt.show()
