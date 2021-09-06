import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

le = LabelEncoder()
lr = LogisticRegression()

def train():
    df = pd.read_json('iris.json')
    df['species'] = le.fit_transform(df[['species']])
    Xtrain,Xtest,ytrain,ytest = train_test_split(df.drop(['species'],axis=1),df[['species']],random_state=14,test_size=0.2)
    lr.fit(Xtrain,ytrain)
    

def prediction(a,b,c,d):
    df = pd.DataFrame([a])
    df[1] = b
    df[2] = c
    df[3] = d
    print(df)
    return le.inverse_transform(lr.predict(df))

train()
print(prediction(10,20,10,15))
