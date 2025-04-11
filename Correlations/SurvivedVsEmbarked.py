import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import math

df = pd.read_csv('train.csv')


df = df[['Survived','Embarked']]


print(df.isna().sum())

df.dropna(inplace=True)

print(df.isna().sum())
print(df.nunique())


contingency_table = pd.crosstab(df['Survived'], df['Embarked'])
print("Contingency Table:")
print(contingency_table)

# 2. Run the Chi-Square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("\nChi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)

# 3. Calculate Cramér's V
n = contingency_table.sum().sum()  # total observation count
min_dim = min(contingency_table.shape) - 1
cramers_v = math.sqrt(chi2 / (n * min_dim))
print("\nCramér's V:", cramers_v)


#+--------------------------------------------------------------+
#|                                                              |
#|    Cramér's V: 0.17261682709984438                           |
#|                                                              |
#|                                                              |
#|                                                              |
#|    P-value: 1.769922284120912e-06                            |
#|                                                              |
#|                                                              |
#+--------------------------------------------------------------+