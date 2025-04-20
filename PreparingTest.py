from joblib import load
import pandas as pd

model = load(r'C:\Users\PC\Desktop\Titanic-Data-Exploration\TitanicLogisticRegression.joblib')

df = pd.read_csv('test.csv')




df['S'] = df['Embarked'].map({'S':1,'C':0,'Q':0})
df['C'] = df['Embarked'].map({'S':0,'C':1,'Q':0})
df['Q'] = df['Embarked'].map({'S':0,'C':0,'Q':1})

df['Sex'] = df['Sex'].map({'male':1,'female':2})

fareMean = df['Fare'].mean()

df['Fare'] = df['Fare'].fillna(fareMean)


x = df[['S','C','Q','Sex','Fare']]

df['Survived'] = model.predict(x)
print(x.isna().sum())
print(df.isna().sum())

df = df[['PassengerId','Survived']]


SecondSubmission = 'SecondSubmission.csv'

df.to_csv(SecondSubmission, index=False)

print(df)