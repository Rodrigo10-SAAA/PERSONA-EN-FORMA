
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


st.write("# Predicción sobre la condición física de una persona")
st.image(
    "ejercicio.jpeg",
    caption="Esta app está diseñada para predecir si una persona estará en forma o no a partir de ciertos parámetros."
)

st.header("Datos de evaluación")

FEATURE_COLS = [
    "age",
    "height_cm",
    "weight_kg",
    "heart_rate",
    "sleep_hours",
    "activity_index",
    "smokes",
    "gender",
]

def user_input_features():
    age = st.number_input("Edad", min_value=1, max_value=100, value=25, step=1)
    height_cm = st.number_input("Altura (cm)", min_value=100, max_value=250, value=170, step=1)
    weight_kg = st.number_input("Peso (kg)", min_value=30, max_value=200, value=70, step=1)
    heart_rate = st.number_input("Ritmo cardiaco (lpm)", min_value=40, max_value=200, value=70, step=1)
    sleep_hours = st.number_input("Horas de sueño", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
    activity_index = st.number_input("Índice de actividad (0-10)", min_value=0, max_value=10, value=5, step=1)
    smokes = st.selectbox("Fuma (0=No, 1=Sí)", [0, 1])
    gender = st.selectbox("Género (0=H, 1=M)", [0, 1])

    data = {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "heart_rate": heart_rate,
        "sleep_hours": sleep_hours,
        "activity_index": activity_index,
        "smokes": smokes,
        "gender": gender,
    }
    return pd.DataFrame(data, index=[0])


df = user_input_features()


data = pd.read_csv("Fitness_Classification_Modificado.csv", encoding="latin-1")

X = data[FEATURE_COLS]
y = data["is_fit"]


classifier = DecisionTreeClassifier(
    max_depth=3,
    criterion="entropy",
    min_samples_leaf=100,
    max_features=4,
    random_state=1615173,
)
classifier.fit(X, y)

st.subheader("Predicción")

df_model = df[FEATURE_COLS]
pred = classifier.predict(df_model)[0]

if pred == 1:
    st.write("LA PERSONA ESTA EN FORMA")
else:
    st.write("LA PERSONA NO ESTA EN FORMA")



 
