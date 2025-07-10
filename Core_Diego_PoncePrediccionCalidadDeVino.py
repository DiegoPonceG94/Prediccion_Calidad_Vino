# Librerías para manejo de datos y visualización
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Librerías para machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Conexión con Google Drive
from google.colab import drive
drive.mount('/content/drive')  # Montamos el Drive para acceder a archivos

# Cargar el archivo CSV desde Google Drive
ruta_archivo = "/content/drive/MyDrive/datasets/WineQT.csv"
vino_df = pd.read_csv(ruta_archivo)

# Exploración básica del dataset
vino_df.info()               # Información general del DataFrame
vino_df.head()               # Primeras filas del dataset
vino_df.isnull().sum()       # Conteo de valores nulos por columna
vino_df.duplicated().sum()   # Total de filas duplicadas
vino_df.dtypes               # Tipos de datos por columna

# Eliminación de columna innecesaria
vino_df.drop(columns=["Id"], inplace=True)  # Eliminamos la columna 'Id'

# Verificación de la estructura final del DataFrame
vino_df.info()

# Separación de variables predictoras y objetivo
caracteristicas = vino_df.drop(columns=["quality"])
objetivo = vino_df["quality"]

# División del dataset en entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    caracteristicas, objetivo, test_size=0.2, random_state=42
)

# Todas las variables predictoras son numéricas
columnas_numericas = caracteristicas.columns.tolist()

# Preprocesamiento para KNN: escalamiento estándar
preprocesador_knn = ColumnTransformer([
    ("numericas", StandardScaler(), columnas_numericas)
])

# Pipeline KNN
modelo_knn = Pipeline([
    ("preparacion", preprocesador_knn),
    ("clasificador", KNeighborsClassifier())
])

# Búsqueda de hiperparámetros para KNN
parametros_knn = {
    "clasificador__n_neighbors": [2, 3, 5, 10]
}

busqueda_knn = GridSearchCV(modelo_knn, parametros_knn, cv=3, scoring="accuracy")
busqueda_knn.fit(X_entrenamiento, y_entrenamiento)

# Evaluación del mejor modelo KNN
mejor_knn = busqueda_knn.best_estimator_
predicciones_knn = mejor_knn.predict(X_prueba)

print("RESULTADOS KNN")
print("Parámetros óptimos:", busqueda_knn.best_params_)
print("Precisión:", accuracy_score(y_prueba, predicciones_knn))

# Entrenar un modelo KNN fijo con 10 vecinos
modelo_knn_10 = Pipeline([
    ("preparacion", preprocesador_knn),
    ("clasificador", KNeighborsClassifier(n_neighbors=10))
])

modelo_knn_10.fit(X_entrenamiento, y_entrenamiento)
predicciones_knn_10 = modelo_knn_10.predict(X_prueba)

# Preprocesamiento para Random Forest
preprocesador_rf = ColumnTransformer([
    ("numericas", "passthrough", columnas_numericas)
])

# Pipeline para Random Forest
modelo_rf = Pipeline([
    ("preparacion", preprocesador_rf),
    ("clasificador", RandomForestClassifier(random_state=42))
])

# Búsqueda de hiperparámetros para Random Forest
parametros_rf = {
    "clasificador__n_estimators": [50, 100, 200]
}

busqueda_rf = GridSearchCV(modelo_rf, parametros_rf, cv=3, scoring="accuracy")
busqueda_rf.fit(X_entrenamiento, y_entrenamiento)

# Evaluación del mejor modelo RF
mejor_rf = busqueda_rf.best_estimator_
predicciones_rf = mejor_rf.predict(X_prueba)

print("\nRESULTADOS RANDOM FOREST")
print("Parámetros óptimos:", busqueda_rf.best_params_)
print("Precisión:", accuracy_score(y_prueba, predicciones_rf))

# Random Forest con 50 árboles
modelo_rf_50 = Pipeline([
    ("preparacion", preprocesador_rf),
    ("clasificador", RandomForestClassifier(n_estimators=50, random_state=42))
])

modelo_rf_50.fit(X_entrenamiento, y_entrenamiento)
predicciones_rf_50 = modelo_rf_50.predict(X_prueba)

# Regresión Logística con escalado
preprocesador_rl = ColumnTransformer([
    ("numericas", StandardScaler(), columnas_numericas)
])

modelo_rl = Pipeline([
    ("preparacion", preprocesador_rl),
    ("clasificador", LogisticRegression(max_iter=1000, random_state=42))
])

modelo_rl.fit(X_entrenamiento, y_entrenamiento)
predicciones_rl = modelo_rl.predict(X_prueba)

# Evaluación de todos los modelos
print("ACCURACY COMPARATIVO")
print("KNN (n=10):", accuracy_score(y_prueba, predicciones_knn_10))
print("Random Forest (n=50):", accuracy_score(y_prueba, predicciones_rf_50))
print("Regresión Logística:", accuracy_score(y_prueba, predicciones_rl))

# -------------------------------
# MATRICES DE CONFUSIÓN
# -------------------------------

# Etiquetas ordenadas
etiquetas = sorted(y_prueba.unique())

# Matriz de confusión: Random Forest
matriz_rf = confusion_matrix(y_prueba, predicciones_rf_50, labels=etiquetas)

plt.figure(figsize=(6, 6))
sns.heatmap(matriz_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=etiquetas, yticklabels=etiquetas, cbar=False)
plt.title("Matriz de Confusión - RF (n=50)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Matriz de confusión: KNN
matriz_knn = confusion_matrix(y_prueba, predicciones_knn_10, labels=etiquetas)

plt.figure(figsize=(6, 6))
sns.heatmap(matriz_knn, annot=True, fmt="d", cmap="Greens",
            xticklabels=etiquetas, yticklabels=etiquetas, cbar=False)
plt.title("Matriz de Confusión - KNN (n=10)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Matriz de confusión: Regresión Logística
matriz_rl = confusion_matrix(y_prueba, predicciones_rl, labels=etiquetas)

plt.figure(figsize=(6, 6))
sns.heatmap(matriz_rl, annot=True, fmt="d", cmap="Oranges",
            xticklabels=etiquetas, yticklabels=etiquetas, cbar=False)
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

#Analisis y comparación de resultados


#Tras entrenar y evaluar los tres modelos (KNN, Random Forest y Regresión Logística), se obtuvieron los siguientes niveles de precisión (accuracy) en el conjunto de prueba:

#K-Nearest Neighbors (KNN, n=10): ~0.58
#Random Forest (n=50 árboles): ~0.64
#Regresión Logística (max_iter=1000): ~0.53

#El modelo que mejor desempeño logró en términos de precisión fue Random Forest, superando tanto a KNN como a la Regresión Logística. Esto era esperable, ya que Random Forest es un modelo de ensamble que suele manejar bien la complejidad y la no linealidad de los datos, además de ser robusto frente al sobreajuste.

#KNN también mostró un rendimiento aceptable, pero con mayor sensibilidad a la elección de hiperparámetros como el número de vecinos. Requiere una buena normalización previa para evitar sesgos por escala, cosa que se manejó con StandardScaler.

#Por su parte, la Regresión Logística mostró el rendimiento más bajo, lo que indica que es posible que el problema no sea linealmente separable, o que la naturaleza multiclase del target esté afectando la eficacia de este modelo.

#Interpretación de las Matrices de Confusión
#Las matrices de confusión indicaron que la mayoría de los errores de predicción se dieron entre clases adyacentes, como predecir un 6 cuando en realidad era un 7, o un 5 en lugar de un 6.

#Esto sugiere que la calidad del vino tiene una cierta continuidad numérica, y que podría incluso explorarse el modelado como problema de regresión o como clasificación ordinal, en lugar de clasificación tradicional multiclase.

#Conclusiones Generales

#El mejor modelo para este conjunto de datos fue Random Forest, que logró el mayor puntaje de accuracy y una matriz de confusión más equilibrada.

#KNN se comporta bien pero requiere una cuidadosa selección de parámetros y un preprocesamiento sólido.

#Regresión Logística no parece ser la mejor opción para este caso, aunque es útil como modelo base.

#En futuros trabajos, se podrían explorar técnicas más avanzadas como:

#XGBoost o LightGBM

#Modelos de clasificación ordinal

#Transformación del problema a regresión

#Técnicas de balanceo de clases si existiera desbalance

#También sería interesante realizar análisis de importancia de variables con modelos de árbol para entender mejor qué características del vino influyen más en su calidad.