import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('train.csv')

df = df[['Survived','Embarked','Fare','Pclass','Sex']]


df.dropna(inplace=True)

df['Sex'] = df['Sex'].map({'male':1,'female':2})




df['S'] = df['Embarked'].map({'S':1,'C':0,'Q':0})
df['C'] = df['Embarked'].map({'S':0,'C':1,'Q':0})
df['Q'] = df['Embarked'].map({'S':0,'C':0,'Q':1})

df = df[['S','C','Q','Survived','Sex','Pclass','Fare']]


print(df['Fare'].median())

df['Group'] = pd.cut(df['Fare'], bins = 80)

matrix = pd.crosstab(df['Survived'],df['Fare'])

plt.boxplot(matrix, patch_artist=True)



plt.show()