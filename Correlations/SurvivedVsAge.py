import pandas as pd
from scipy.stats import pointbiserialr as pbs

df = pd.read_csv('train.csv')

df = df[['Survived','Age']]

print(df)



print(df.isna().sum())


df = df.dropna()


correlation, pValue = pbs(df['Survived'], df['Age'])

print(correlation)
print(pValue)

# +--------------------------------------+
# |                                      |
# |  Correlation: -0.07722109457217767   |
# |  P Value: 0.039124654013482654       |
# |                                      |
# +--------------------------------------+