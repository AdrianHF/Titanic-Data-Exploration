import pandas as pd

df = pd.read_csv('train.csv')


df = df[['Survived','Ticket']]


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df)


# +------------------+
# | Not worth it yet |
# +------------------+