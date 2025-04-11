import pandas as pd
from scipy.stats import pearsonr 


df = pd.read_csv('train.csv')


df = df[['Survived','Sex']]

df['Sex'] = df['Sex'].map({'male':1,'female':0})


print(df)

r, p_value = pearsonr (df['Sex'], df['Survived'])

print(r)
print(p_value)



#+------------------------------------+
#|                                    |
#| Correlation: -0.5433513806577551   |
#|                                    |
#| P-Value: 1.406066130880276e-69     |
#|                                    |
#+------------------------------------+