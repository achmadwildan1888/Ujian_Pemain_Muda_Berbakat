import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv('data.csv')
df=df.fillna(np.NaN)

df['Target']=0
df['Target_name']='Non-Target'


df['Target'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]=1
df['Target_name'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]='Target'

x=df.loc[:,['Age','Overall','Potential']]
y=df['Target']

k = round(len(x) ** .5)
if((k%2) == 0):
    k=k+1
else:
    k=k
    
x=df.loc[:,['Age','Overall','Potential']]
y=df['Target']
from sklearn.model_selection import KFold
k = KFold(n_splits = 4) 

skorlogreg=[]
skorsvc=[]
skorrandomfor=[]

x=df.loc[:,['Age','Overall','Potential']]
y=df['Target']
kf=KFold(n_splits = 3)
for train_index,test_index in kf.split(x):
    x_train=x.iloc[train_index]
    y_train=y[train_index]
  
    
print("hasil logreg = ",cross_val_score(
    LogisticRegression(),
    x_train,
    y_train).mean())

print("hasil SVM =",cross_val_score(
    SVC(gamma='auto'),
     x_train,
     y_train).mean())
      
print("hasil random forest =",cross_val_score(
    RandomForestClassifier(n_estimators=100),
     x_train,
     y_train).mean())