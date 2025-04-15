import pandas as pd

df = pd.read_csv('test.csv')


print(df)

df['Sex'] = df['Sex'].map({'male':1,'female':2})




df['S'] = df['Embarked'].map({'S':1,'C':0,'Q':0})
df['C'] = df['Embarked'].map({'S':0,'C':1,'Q':0})
df['Q'] = df['Embarked'].map({'S':0,'C':0,'Q':1})


print(df.isna().sum())

df = df[['S','C','Q','Sex','Fare']]

print(df)