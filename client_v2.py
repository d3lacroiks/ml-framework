import requests
url = 'https://ml-framework-api.onrender.com/predict'
data = {
    "edad": 35,
    "peso_inicial_kg": 78,
    "altura_cm": 169,
    "actividad_fisica_semanal": 1,
    "horas_cardio_semana": 1,
    "horas_fuerza_semana": 0,
    "horas_hiit_semana": 0,
    "tiene_pesas": 0,
    "tiene_bandas": 1,
    "espacio_pequeno": 1,
    "calorias_diarias": 2000,
    "horas_sueno": 8,
    "autoestima_inicial": 9,
    "estres_inicial": 2,
    "sexo_M": 0,
    "tipo_dieta_cetogenica": 1
}
resp = requests.post(url, json=data).json()
print("Probabilidad:", resp['prob'])
print("Factores:", resp['factores'])
print("Recomendación:", resp['recomendacion'])