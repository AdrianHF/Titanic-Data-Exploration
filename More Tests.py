import pandas as pd
from scipy.stats import pointbiserialr


df = pd.read_csv('train.csv')


df = df[['Survived','Name']]


df['Length'] = df['Name'].apply(len)



print(df['Length'])

binaryVar = df['Survived']
contVar = df['Length']

corr, pValue = pointbiserialr(binaryVar, contVar)

print(corr, pValue)


df['Name'] = df['Name'].str.replace(r'\(.*?\)','', regex=True)

print(df)


df['Length'] = df['Name'].apply(len)

print(df)

binaryVar = df['Survived']
contVar = df['Length']

corr, pValue = pointbiserialr(binaryVar, contVar)

print(corr, pValue)