import pandas as pd
import numpy as np

# Cargar datos
data = pd.read_csv('train.csv')

# Limpieza
# 1. Eliminar columnas no relevantes
data_clean = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# 2. Manejar valores faltantes en Age
median_ages = data_clean.groupby(['Pclass', 'Sex'])['Age'].median()
data_clean['Age'] = data_clean.apply(
    lambda row: median_ages[row['Pclass']][row['Sex']] if pd.isnull(row['Age']) else row['Age'], 
    axis=1
)

# 3. Manejar valores faltantes en Fare (solo 1 caso)
data_clean['Fare'] = data_clean['Fare'].fillna(data_clean['Fare'].median())

# 4. Codificar Sex
data_clean['Sex'] = data_clean['Sex'].map({'male': 0, 'female': 1})

# 5. Crear nuevas características
data_clean['FamilySize'] = data_clean['SibSp'] + data_clean['Parch']
data_clean['IsAlone'] = (data_clean['FamilySize'] == 0).astype(int)

# 6. Eliminar outliers
data_clean = data_clean[data_clean['Fare'] <= 300]
data_clean = data_clean[(data_clean['Age'] > 0) & (data_clean['Age'] < 80)]

# 7. Crear variables dummy para Pclass
data_clean = pd.get_dummies(data_clean, columns=['Pclass'], prefix='Pclass')

# 8. Eliminar columnas originales que ya no necesitamos
data_clean = data_clean.drop(['SibSp', 'Parch'], axis=1)

# Verificar que no hay valores faltantes
print(data_clean.isnull().sum())

# Mostrar las primeras filas del dataset limpio
print(data_clean.head())

data_clean.to_csv('titanic_clean.csv', index=False)
