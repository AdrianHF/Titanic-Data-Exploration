import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('titanic_clean.csv')

print(df)

x = df[['Sex','Age','Fare','FamilySize','IsAlone','Pclass_1', 'Pclass_2','Pclass_3']]
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.1, 
    random_state=42,  # Para reproducibilidad
    stratify=y  # Mantiene proporción de clases en ambos conjuntos
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(
    max_iter=1000,  # Aumentar iteraciones
    solver='lbfgs',  # Buen solver por defecto
    class_weight='balanced',  # Opcional si hay desbalance de clases
    random_state=42
)


model.fit(X_train_scaled, y_train)

# 5. Evaluar el modelo
y_pred = model.predict(X_test_scaled)

print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

