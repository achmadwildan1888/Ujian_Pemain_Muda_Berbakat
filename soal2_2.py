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
    
model1=LogisticRegression(multi_class='auto',solver='liblinear')
model2=RandomForestClassifier(n_estimators=100)
model3 = SVC(gamma = 'auto')


print("Skor Logistic Regression: ",round(cross_val_score(model1,x,y,cv=3).mean()*100),' %')
print("Skor Random Forest: ",round(cross_val_score(model2,x,y,cv=3).mean()*100),' %')
print("Skor SVM: ",round(cross_val_score(model3,x,y,cv=3).mean()*100),' %')