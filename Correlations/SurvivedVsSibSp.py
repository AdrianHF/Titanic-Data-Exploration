import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr as pbs

df = pd.read_csv('train.csv')

df = df[['Survived','SibSp']]

print(df)

uniqueVCount = df['SibSp'].nunique()

print(uniqueVCount)

print(df['SibSp'].value_counts())


df['SibSp'].value_counts().plot(kind='bar')



plt.xlabel('SibSp')
plt.ylabel('Count')


heatmap_data = pd.crosstab(df['SibSp'], df['Survived'])
print(heatmap_data)

sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="crest")
plt.title("Frequency of Category Pairs")
plt.show()


correlation, pValue = pbs(df['Survived'], df['SibSp']) 

print(correlation)

print(pValue)