import requests
import json

url = 'http://localhost:5000/predict'
data = {
    "edad": 35,
    "peso_inicial_kg": 90,
    "altura_cm": 170,
    "horas_ejercicio_semana": 6,
    "calorias_diarias": 1500,
    "sexo_M": 1
}

resp = requests.post(url, json=data)
print("Respuesta (peso):", resp.json())