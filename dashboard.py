import streamlit as st
import requests
import json
import joblib

# Cargar configuración activa localmente (opcional, para mostrar el target)
with open('model_registry/active_version.txt', 'r') as f:
    version = f.read().strip()
model_path = f'model_registry/model_{version}.pkl'
model_data = joblib.load(model_path)
config = model_data.get('config', {})
target = config.get('target', 'desconocido')

st.set_page_config(page_title="Framework ML", layout="wide")
st.title(f"🔮 Predicción de {target}")
st.markdown(f"Modelo activo: `{version}`")

# URL de la API (ajusta si tu URL en Render cambia)
API_URL = "https://ml-framework-api.onrender.com/predict"

with st.form("formulario"):
    col1, col2 = st.columns(2)
    with col1:
        edad = st.number_input("Edad", 18, 80, 35)
        peso = st.number_input("Peso inicial (kg)", 40, 200, 95)
        altura = st.number_input("Altura (cm)", 130, 210, 165)
        ejercicio = st.number_input("Horas ejercicio/semana", 0, 14, 1)
        calorias = st.number_input("Calorías diarias", 1000, 3500, 2300)
    with col2:
        sueno = st.number_input("Horas de sueño", 4, 10, 5)
        autoestima = st.slider("Autoestima (1-10)", 1, 10, 3)
        estres = st.slider("Estrés (1-10)", 1, 10, 9)
        sexo = st.radio("Sexo", ["M", "F"], index=0)
        dieta = st.selectbox("Tipo de dieta", ["balanceada", "baja_carb", "cetogenica", "mediterranea"])

    submitted = st.form_submit_button("Predecir")

if submitted:
    payload = {
        "edad": edad,
        "peso_inicial_kg": peso,
        "altura_cm": altura,
        "actividad_fisica_semanal": ejercicio / 2,
        "horas_ejercicio_semana": ejercicio,
        "calorias_diarias": calorias,
        "horas_sueno": sueno,
        "autoestima_inicial": autoestima,
        "estres_inicial": estres,
        "sexo_M": 1 if sexo == "M" else 0,
        "tipo_dieta_balanceada": 1 if dieta == "balanceada" else 0,
        "tipo_dieta_cetogenica": 1 if dieta == "cetogenica" else 0,
        "tipo_dieta_mediterranea": 1 if dieta == "mediterranea" else 0,
        "tipo_dieta_baja_carb": 1 if dieta == "baja_carb" else 0
    }

    with st.spinner("Consultando API..."):
        resp = requests.post(API_URL, json=payload).json()

    st.metric(f"Probabilidad de {target}", f"{resp['prob']:.2%}")
    st.write("**Factores SHAP:**")
    st.json(resp['factores'])