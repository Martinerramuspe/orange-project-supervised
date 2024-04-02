import requests
from io import StringIO
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from joblib import load  # Importar la función load

# Creamos preprocesador, en vez de cargarlo:-----------------------------------------------------------------------


# Función para filtrar columnas
def filtrar_columnas(df):
    columnas_a_mantener = [
        'International plan', 'Voice mail plan', 'Number vmail messages',
        'Total day minutes', 'Total day charge', 'Total eve minutes',
        'Total eve charge', 'Total night minutes', 'Total night charge',
        'Total intl minutes', 'Total intl calls'
    ]
    return df[columnas_a_mantener]

# Función que transforma 'International plan' a int.
def mapear_international_plan(df):
    df_copy = df.copy()
    df_copy['International plan'] = df_copy['International plan'].map({'Yes': 1, 'No': 0})
    return df_copy

# Función que transforma 'Voice mail plan' a int.
def mapear_voice_mail_plan(df):
    df_copy = df.copy()
    df_copy['Voice mail plan'] = df_copy['Voice mail plan'].map({'Yes': 1, 'No': 0})
    return df_copy

# Función para eliminar outliers
def eliminar_outliers(df):
    columns_to_check = ['Number vmail messages', 'Total day minutes',
                        'Total day charge', 'Total eve minutes',
                        'Total eve charge', 'Total night minutes',
                        'Total night charge', 'Total intl minutes',
                        'Total intl calls']

    df_copy = df.copy()

    for column in columns_to_check:
        q1 = df_copy[column].quantile(0.25)
        q3 = df_copy[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df_copy = df_copy.drop(df_copy[(df_copy[column] < lower_bound) | (df_copy[column] > upper_bound)].index)

    return df_copy

# Función para normalizar variables
def norma_variables(df):
    columns_to_normalize = ['Number vmail messages', 'Total day minutes',
                            'Total day charge', 'Total eve minutes',
                            'Total eve charge', 'Total night minutes',
                            'Total night charge', 'Total intl minutes',
                            'Total intl calls']

    df_copy = df.copy()
    scaler = MinMaxScaler()
    df_copy[columns_to_normalize] = scaler.fit_transform(df_copy[columns_to_normalize])
    return df_copy

# Creación de transformadores
Best_caracter = FunctionTransformer(filtrar_columnas)
Mapeo_01 = FunctionTransformer(mapear_international_plan)
Mapeo_02 = FunctionTransformer(mapear_voice_mail_plan)
Outliers_drop = FunctionTransformer(eliminar_outliers)
Normalizacion = FunctionTransformer(norma_variables)

# Creación del pipeline de preprocesamiento
Prepro01 = Pipeline(steps=[
    ("Best_features", Best_caracter),
    ("Mapeo_01", Mapeo_01),
    ("Mapeo_02", Mapeo_02),
    ("Outliers_drop", Outliers_drop),
    ("Normalizacion", Normalizacion),
])




# Cargar el modelo desde el archivo .joblib-----------------------------------------------------------------------
modelo = load('RandomForestClassifier.joblib')

# Función para hacer la predicción
def predecir(data):
    # Preprocesar los datos
    data_preprocesada = Prepro01.transform(data)
    # Realizar la predicción con el modelo
    prediction = modelo.predict(data_preprocesada)
    return prediction[0]

# Creacion de titulos, subtitulos, e insercion de imagen.
st.title('MODELO PREDICTIVO')
st.image("https://github.com/Martinerramuspe/04-ADJUNTOS/raw/main/orange.png", use_column_width=True)
st.write('Con esta herramienta podemos saber si se va a dar de baja o no un cliente en función a su historial dentro de la empresa.')

# Definir los datos de entrada.
X = {
    'State': 'NY', # valor fijo.
    'Account length': 150,# valor fijo.
    'Area code': 212,# valor fijo.
    'International plan': 'Yes',
    'Voice mail plan': 'No',
    'Number vmail messages': 0,
    'Total day minutes': 300.5,
    'Total day calls': 120,# valor fijo.
    'Total day charge': 50.75,
    'Total eve minutes': 250.3,
    'Total eve calls': 80,# valor fijo.
    'Total eve charge': 21.15,
    'Total night minutes': 200.2,
    'Total night calls': 100,# valor fijo.
    'Total night charge': 9.01,
    'Total intl minutes': 15.5,
    'Total intl calls': 4,
    'Total intl charge': 4.2,# valor fijo.
    'Customer service calls': 2,# valor fijo.
    'Churn': False# valor fijo.
}

# Creamos sliders para los inputs.
input_Number_vmail_messages = st.slider('Number vmail messages', min_value=0, max_value=70, value=X['Number vmail messages'])
input_Total_day_minutes = float(st.slider('Total day minutes', min_value=0, max_value=400, value=int(X['Total day minutes'])))
input_Total_day_charge = float(st.slider('Total day charge', min_value=0, max_value=80, value=int(X['Total day charge'])))#0 a 80.
input_Total_eve_minutes = float(st.slider('Total eve minutes', min_value=0, max_value=400, value=int(X['Total eve minutes'])))# 0 a 400.
input_Total_eve_charge = float(st.slider('Total eve charge', min_value=0, max_value=40, value=int(X['Total eve charge'])))# 0 a 40.
input_Total_eve_charge = float(st.text_input('Total eve charge', value=str(X['Total eve charge']))) #0 a 40.
input_Total_night_minutes = float(st.text_input('Total night minutes', value=str(X['Total night minutes']))) #0 a 450.
input_Total_night_charge = float(st.text_input('Total night charge', value=str(X['Total night charge']))) #0 a 20.
input_Total_intl_minutes = float(st.text_input('Total intl minutes', value=str(X['Total intl minutes']))) #0 a 30.
input_Total_intl_calls = int(st.text_input('Total intl calls', value=str(X['Total intl calls']))) # 0 a 30.


# Actualizar los valores de entrada.
X.update({
    'Number vmail messages': input_Number_vmail_messages,
    'Total day minutes': input_Total_day_minutes,
    'Total day charge': input_Total_day_charge,
    'Total eve minutes': input_Total_eve_minutes,
    'Total eve charge': input_Total_eve_charge,
    'Total night minutes': input_Total_night_minutes,
    'Total night charge': input_Total_night_charge,
    'Total intl minutes': input_Total_intl_minutes,
    'Total intl calls': input_Total_intl_calls
})

# Creamos radio para 'International plan' y 'Voice mail plan'.
input_international_plan = st.radio('International plan', ['Yes', 'No'], index=0 if X['International plan'] == 'Yes' else 1)
input_voice_mail_plan = st.radio('Voice mail plan', ['Yes', 'No'], index=0 if X['Voice mail plan'] == 'Yes' else 1)

# Actualizar los valores de 'International plan' y 'Voice mail plan'
X.update({
    'International plan': input_international_plan,
    'Voice mail plan': input_voice_mail_plan
})

# Convertir los datos recolectados en "X" a un DataFrame para poder meter en modelo.
X_df = pd.DataFrame([X])

# Realizar la predicción cuando se presiona el botón
if st.button('Predecir'):
    X_df_preprocesado = Prepro01.transform(X_df)  # Preprocesar los datos de entrada
    prediction = predecir(X_df_preprocesado)  # Realizar la predicción con el modelo
    st.write('La predicción es:', prediction)
