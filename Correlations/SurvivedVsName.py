#The purpose for this script is to find the correlation between the variable 
#'Name' and 'Survived' in our dataset, we will use the length of the name as a way
#to quantify the variable name

import pandas as pd
from scipy.stats import pointbiserialr as pbs

df = pd.read_csv('train.csv')



#1.1 - Working with our two columns

df = df[['Survived','Name']]


#1.2 - Meassuring the length of the names in our name column

df['Length'] =  df['Name'].apply(len)


# 1.3 - Meassuring correlation between the length of the names and the Survived 
# status datapoints using Point-Biserial Correlation. 

# The Point-Biserial Correlation is a statistical measure used to determine 
# the relationship between a binary variable (e.g., 0/1, yes/no) and a continuous variable.

binaryVar = df['Survived']
contVar = df['Length']

corr, pValue = pbs(binaryVar, contVar)

print('Correlation: ',corr)
print('P Value: ',pValue)


#+--------------------------------------------------------------+
#| Correlation:  0.3323495344232765                             |
#|                                                              |
#| (1 indicates a perfect positive correlation.                 |
#| -1 indicates a perfect negative correlation.                 |
#| 0 suggests no correlation.)                                  |
#|                                                              |
#| P Value:  2.0267950663436688e-24                             |
#|                                                              |
#| Typically, if the p-value is less than 0.05, the correlation |
#| is considered statistically significant—this means the       |
#| relationship observed is likely not random, and there is     |
#| evidence of a true association between the variables.        |
#+--------------------------------------------------------------+

# --------------------------------------------------------------------------------------


# Many of the 'Name' datapoints include a parentheses with a different name, 
# we will find the correlation between the names and the 'Survived' status
# without the parentheses information



df['Name'] = df['Name'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

df['Length'] =  df['Name'].apply(len)
binaryVar = df['Survived']
contVar = df['Length']

corr, pValue = pbs(binaryVar, contVar)

print('Correlation: ',corr)
print('P Value: ',pValue)

#+--------------------------------------------------------------+
#| Correlation:  0.09937304899666002                            |
#|                                                              |
#|                                                              |
#| P Value:  0.002983462889232356                               |
#|                                                              |
#+--------------------------------------------------------------+



print(df)