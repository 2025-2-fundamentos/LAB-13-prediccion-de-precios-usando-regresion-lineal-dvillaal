import os
import gzip
import json
import pickle
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def limpiar_datos(df):
    datos = df.copy()
    datos = datos.dropna()
    datos["Age"] = 2021 - datos["Year"]
    datos = datos.drop(columns=["Year", "Car_Name"])
    return datos


def crear_modelo():
    columnas_categoricas = ["Fuel_Type", "Selling_type", "Transmission"]
    columnas_numericas = ["Selling_Price", "Driven_kms", "Age", "Owner"]

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("scaler", MinMaxScaler(), columnas_numericas),
        ],
        remainder="passthrough",
    )

    seleccion_caracteristicas = SelectKBest(score_func=f_classif)

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("seleccion", seleccion_caracteristicas),
            ("regresor", LinearRegression()),
        ]
    )

    return modelo


def optimizar_hiperparametros(modelo, n_splits, x_entrenamiento, y_entrenamiento, puntuacion):
    buscador = GridSearchCV(
        estimator=modelo,
        param_grid={"seleccion__k": range(1, 13)},
        cv=n_splits,
        refit=True,
        scoring=puntuacion,
    )
    buscador.fit(x_entrenamiento, y_entrenamiento)
    return buscador


def calcular_metricas(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    y_entrenamiento_pred = modelo.predict(x_entrenamiento)
    y_prueba_pred = modelo.predict(x_prueba)

    metricas_entrenamiento = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_entrenamiento, y_entrenamiento_pred),
        "mse": mean_squared_error(y_entrenamiento, y_entrenamiento_pred),
        "mad": median_absolute_error(y_entrenamiento, y_entrenamiento_pred),
    }

    metricas_prueba = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_prueba, y_prueba_pred),
        "mse": mean_squared_error(y_prueba, y_prueba_pred),
        "mad": median_absolute_error(y_prueba, y_prueba_pred),
    }

    return metricas_entrenamiento, metricas_prueba


def guardar_modelo(modelo):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as archivo:
        pickle.dump(modelo, archivo)


def guardar_metricas(lista_metricas):
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as archivo:
        for metrica in lista_metricas:
            linea = json.dumps(metrica)
            archivo.write(linea + "\n")


archivo_prueba = "files/input/test_data.csv.zip"
archivo_entrenamiento = "files/input/train_data.csv.zip"

with zipfile.ZipFile(archivo_prueba, "r") as zip_prueba:
    with zip_prueba.open("test_data.csv") as archivo:
        df_prueba = pd.read_csv(archivo)

with zipfile.ZipFile(archivo_entrenamiento, "r") as zip_train:
    with zip_train.open("train_data.csv") as archivo:
        df_entrenamiento = pd.read_csv(archivo)

df_prueba = limpiar_datos(df_prueba)
df_entrenamiento = limpiar_datos(df_entrenamiento)

x_entrenamiento = df_entrenamiento.drop("Present_Price", axis=1)
y_entrenamiento = df_entrenamiento["Present_Price"]
x_prueba = df_prueba.drop("Present_Price", axis=1)
y_prueba = df_prueba["Present_Price"]

pipeline_modelo = crear_modelo()
pipeline_modelo = optimizar_hiperparametros(
    pipeline_modelo,
    n_splits=10,
    x_entrenamiento=x_entrenamiento,
    y_entrenamiento=y_entrenamiento,
    puntuacion="neg_mean_absolute_error",
)

guardar_modelo(pipeline_modelo)

metricas_train, metricas_test = calcular_metricas(
    pipeline_modelo,
    x_entrenamiento,
    y_entrenamiento,
    x_prueba,
    y_prueba,
)

guardar_metricas([metricas_train, metricas_test])
