import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('train.csv')

df = df[['Survived','Embarked','Fare','Pclass','Sex','Name']]


df.dropna(inplace=True)

df['Sex'] = df['Sex'].map({'male':1,'female':2})


uValues = df['Embarked'].unique()

print(uValues)

df['S'] = df['Embarked'].map({'S':1,'C':0,'Q':0})
df['C'] = df['Embarked'].map({'S':0,'C':1,'Q':0})
df['Q'] = df['Embarked'].map({'S':0,'C':0,'Q':1})

df = df[['S','C','Q','Survived','Sex','Pclass','Fare']]





x = df[['S','C','Q','Sex','Pclass','Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



model = LogisticRegression(max_iter=10000)


model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get a detailed classification report
print(classification_report(y_test, y_pred))

