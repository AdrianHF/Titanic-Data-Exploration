import pandas as pd
import numpy as n

df = pd.read_csv('train.csv')

df = df[['Survived','Cabin']]

print(df.isna().sum())


df = df.dropna()

print(df.isna().sum())

print(df.nunique())

# +--------------+
# | Not worth it |
# +--------------+