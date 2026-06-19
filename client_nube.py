import requests

url = 'https://ml-framework-api.onrender.com/predict'
data = {
    "edad": 35,
    "peso_inicial_kg": 95,
    "altura_cm": 165,
    "actividad_fisica_semanal": 1,
    "horas_ejercicio_semana": 1,
    "calorias_diarias": 2300,
    "horas_sueno": 5,
    "autoestima_inicial": 3,
    "estres_inicial": 9,
    "sexo_M": 1,
    "tipo_dieta_balanceada": 1,
    "tipo_dieta_cetogenica": 0,
    "tipo_dieta_mediterranea": 0
}

resp = requests.post(url, json=data)
print(resp.json())