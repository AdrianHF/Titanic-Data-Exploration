import pandas as pd
from scipy.stats import pointbiserialr as pbs

df = pd.read_csv('train.csv')


df = df[['Survived','Parch']]

pd.set_option('display.max_rows', None)


print(df.nunique())


correlation, pValue = pbs(df['Survived'],df['Parch'])

print(correlation)
print(pValue)

# +------------------+
# | Not worth it yet |
# +------------------+