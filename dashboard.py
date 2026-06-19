import streamlit as st
import requests

# Título
st.set_page_config(page_title="Predicción Inteligente", layout="wide")
st.title("🔮 Predicción de éxito en programa de pérdida de peso")

# URL de la API de Render (cambia si es necesario)
API_URL = "https://ml-framework-api.onrender.com/predict"

st.markdown("Completa los datos del paciente para obtener una predicción personalizada y recomendaciones automáticas.")

with st.form("formulario"):
    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.number_input("Edad", 18, 80, 35)
        peso = st.number_input("Peso inicial (kg)", 40, 200, 78)
        altura = st.number_input("Altura (cm)", 130, 210, 169)

    with col2:
        cardio = st.number_input("Horas de cardio/semana", 0, 14, 1)
        fuerza = st.number_input("Horas de fuerza/semana", 0, 8, 0)
        hiit = st.number_input("Horas de HIIT/semana", 0, 4, 0)

    with col3:
        sueno = st.number_input("Horas de sueño", 4, 10, 7)
        autoestima = st.slider("Autoestima (1-10)", 1, 10, 5)
        estres = st.slider("Estrés (1-10)", 1, 10, 5)

    col4, col5 = st.columns(2)
    with col4:
        sexo = st.radio("Sexo", ["M", "F"], index=0)
        dieta = st.selectbox("Tipo de dieta", ["balanceada", "baja_carb", "cetogenica", "mediterranea"])
    with col5:
        calorias = st.number_input("Calorías diarias", 1000, 3500, 2000)
        pesas = st.checkbox("¿Tiene pesas en casa?")
        bandas = st.checkbox("¿Tiene bandas de resistencia?")
        espacio = st.selectbox("Espacio disponible", ["pequeno", "mediano", "grande"])

    submitted = st.form_submit_button("Predecir")

if submitted:
    payload = {
        "edad": edad,
        "peso_inicial_kg": peso,
        "altura_cm": altura,
        "actividad_fisica_semanal": (cardio + fuerza + hiit) / 3,  # aproximado
        "horas_cardio_semana": cardio,
        "horas_fuerza_semana": fuerza,
        "horas_hiit_semana": hiit,
        "tiene_pesas": int(pesas),
        "tiene_bandas": int(bandas),
        "espacio_pequeno": 1 if espacio == "pequeno" else 0,
        "espacio_mediano": 1 if espacio == "mediano" else 0,
        "espacio_grande": 1 if espacio == "grande" else 0,
        "calorias_diarias": calorias,
        "horas_sueno": sueno,
        "autoestima_inicial": autoestima,
        "estres_inicial": estres,
        "sexo_M": 1 if sexo == "M" else 0,
        "tipo_dieta_balanceada": 1 if dieta == "balanceada" else 0,
        "tipo_dieta_baja_carb": 1 if dieta == "baja_carb" else 0,
        "tipo_dieta_cetogenica": 1 if dieta == "cetogenica" else 0,
        "tipo_dieta_mediterranea": 1 if dieta == "mediterranea" else 0
    }

    with st.spinner("Consultando a la inteligencia artificial..."):
        try:
            resp = requests.post(API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            st.success("Predicción generada")
            st.metric("Probabilidad de éxito", f"{data['prob']:.2%}")

            st.subheader("🔎 Factores que más influyen")
            st.json(data['factores'])

            st.subheader("💡 Recomendación personalizada")
            st.info(data['recomendacion'])

        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")