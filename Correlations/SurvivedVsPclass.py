import pandas as pd
import scipy.stats as stats
import numpy as np

df = pd.read_csv('train.csv')

uniqueVCount = df['Pclass'].nunique()


print(uniqueVCount)

contingency_table = pd.crosstab(df['Survived'], df['Pclass'])



# Calculate Cramér’s V
chi2, _, _, _ = stats.chi2_contingency(contingency_table)
n = contingency_table.sum().sum()
phi2 = chi2 / n
r, k = contingency_table.shape
cramers_v = np.sqrt(phi2 / min(k-1, r-1))

print(f"Cramér’s V: {cramers_v:.3f}")


chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
print(f"P-value: {p_value:.100f}")