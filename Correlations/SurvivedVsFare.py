import pandas as pd
from scipy.stats import pointbiserialr as pbs

df = pd.read_csv('train.csv')

df = df[['Survived','Fare']]


print(df.nunique())


pd.set_option('display.max_rows',None)

correlation, pValue = pbs(df['Survived'],df['Fare'])

print(correlation)
print(pValue)

# +------------------------------------+
# |                                    |
# |  Correlation: 0.2573065223849623   |
# |                                    |
# |  P Value: 6.1201893419245925e-15   |
# |                                    |
# +------------------------------------+


