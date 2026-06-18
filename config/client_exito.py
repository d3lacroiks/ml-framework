import requests
url = 'http://localhost:5000/predict'
data = {
    "edad": 28,
    "peso_inicial_kg": 80,
    "altura_cm": 170,
    "actividad_fisica_semanal": 6,
    "horas_ejercicio_semana": 8,
    "calorias_diarias": 1500,
    "horas_sueno": 8,
    "autoestima_inicial": 9,
    "estres_inicial": 2,
    "sexo_M": 0,
    "tipo_dieta_cetogenica": 1,
    "tipo_dieta_mediterranea": 0,
    "tipo_dieta_balanceada": 0
}
resp = requests.post(url, json=data)
print(resp.json())