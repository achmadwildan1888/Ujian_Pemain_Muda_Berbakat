
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


df=pd.read_csv('data.csv')
df=df.fillna(np.NaN)
df['Target']=0
df['Target_name']='Non-Target'

df['Target'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]=1
df['Target_name'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]='Target'

x=df.loc[:,['Age','Overall','Potential']]
y=df['Target_name']

logreg=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=1)
logreg.fit(x_train,y_train)

dfTest=pd.read_csv('pemainindo.csv')
nilai=dfTest.iloc[:,1:]
dfTest['Target']=logreg.predict(nilai)
print(dfTest)