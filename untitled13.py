import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.write('# PREDICCIÓN SOBRE LA CONDICIÓN FÍSICA DE UNA PERSONA')
st.image(
    "ejercicio.jpeg",
    caption="Esta app está diseñada para predecir si una persona estará en forma o no a partir de ciertos parámetros."
)

st.header('Datos de evaluación')

def user_input_features():
    age = st.number_input('EDAD:', min_value=1, max_value=100, value=18, step=1)
    gender = st.number_input('GÉNERO(H=0, F=1):', min_value=0, max_value=1, value=0, step=1)
    height_cm = st.number_input('ALTURA(cm):', min_value=100, max_value=250, value=170, step=1)
    weight_kg = st.number_input('PESO(kg):', min_value=30, max_value=200, value=70, step=1)
    heart_rate = st.number_input('RITMO CARDIACO (40-200):', min_value=40, max_value=200, value=70, step=1)
    sleep_hours = st.number_input('HORAS DE SUEÑO:', min_value=0.0, max_value=20.0, value=8.0, step=0.5)
    activity_index = st.number_input('ÍNDICE DE ACTIVIDAD(0-10)', min_value=0, max_value=10, value=5, step=1)
    smokes = st.number_input('FUMA(NO = 0, SI =1)', min_value=0, max_value=1, value=0, step=1)

    user_input_data = {
        'age': age,
        'gender': gender,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'heart_rate': heart_rate,
        'sleep_hours': sleep_hours,
        'activity_index': activity_index,
        'smokes': smokes
    }
    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

data = pd.read_csv('Fitness_Classification_Modificado.csv', encoding='latin-1')
X = data.drop(columns='is_fit')
Y = data['is_fit']

classifier = DecisionTreeClassifier(
    max_depth=3,
    criterion='entropy',
    min_samples_leaf=100,
    max_features=4,
    random_state=1615173
)
classifier.fit(X, Y)

prediction = classifier.predict(df)[0]

st.subheader('Predicción')
if prediction == 0:
    st.write('NO ESTÁ EN FORMA')
elif prediction == 1:
    st.write('ESTÁ EN FORMA')
else:
    st.write('SIN PREDICCIÓN')
