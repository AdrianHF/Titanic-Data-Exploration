import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tpot import TPOTClassifier

df = pd.read_csv('train.csv')

df = df[['Survived','Embarked','Sex','Fare']]






df.dropna(inplace=True)

df['Sex'] = df['Sex'].map({'male':1,'female':2})


uValues = df['Embarked'].unique()


print(uValues)

df['S'] = df['Embarked'].map({'S':1,'C':0,'Q':0})
df['C'] = df['Embarked'].map({'S':0,'C':1,'Q':0})
df['Q'] = df['Embarked'].map({'S':0,'C':0,'Q':1})

df = df[['S','C','Q','Survived','Sex','Fare']]

print(df)


x = df[['S','C','Q','Sex','Fare']]
y = df['Survived']

model = LogisticRegression()
model.fit(x,y)

# Predecir en datos de prueba
#y_pred = model.predict()

# Evaluar el rendimiento del modelo

#Logistic Regression
