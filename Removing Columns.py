import pandas as pd

# 1.1 - Load the dataset from the CSV file  
df = pd.read_csv("train.csv")


# 1.2 - Check for missing values in each column  
print(df.isnull().sum())

# 1.3 - Remove the 'Cabin' column due to excessive missing data (687 out of 891 entries)  
df = df.drop('Cabin', axis=1)

# 1.4 - Verify remaining missing values after column removal  
print(df.isnull().sum())

# 1.5 - Drop the two rows with missing 'Embarked' values  
df = df.dropna(subset=['Embarked'])

# 1.6 - Verifying empty values one more time
print(df.isnull().sum())


