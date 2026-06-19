import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import shap
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Cargar el modelo activo
with open('model_registry/active_version.txt', 'r') as f:
    version = f.read().strip()
model_path = f'model_registry/model_{version}.pkl'
data = joblib.load(model_path)
modelo = data['modelo']
features = data['features']
config = data.get('config', {})
target = config.get('target', 'desconocido')
print(f"Modelo cargado: {model_path}")

# Explainer SHAP (lo usamos en la predicción)
explainer = shap.TreeExplainer(modelo)

def generar_recomendacion(entrada, shap_vals, target_name):
    """
    Genera recomendaciones textuales basadas en los valores de entrada
    y en los factores de riesgo identificados.
    """
    rec = []
    # Cardio
    if entrada.get('horas_cardio_semana', 0) < 2:
        rec.append("Aumenta el cardio a 3‑4 horas/semana. En casa: saltar la cuerda, caminar en el sitio, bailar, seguir videos de aeróbicos o subir/bajar escaleras.")
    # Fuerza
    if entrada.get('horas_fuerza_semana', 0) < 1.5:
        if entrada.get('tiene_pesas', 0) == 1:
            rec.append("Haz 2‑3 horas de fuerza con mancuernas: sentadillas con peso, press de hombros, remo inclinado, curl de bíceps y peso muerto con mancuernas.")
        elif entrada.get('tiene_bandas', 0) == 1:
            rec.append("Usa bandas de resistencia para ejercicios de fuerza: remo, press de pecho, sentadillas con banda y curl de bíceps.")
        else:
            rec.append("Realiza calistenia 3 veces/semana: flexiones, zancadas, sentadillas sin peso, planchas y burpees.")
    # HIIT
    if entrada.get('horas_hiit_semana', 0) < 0.5 and entrada.get('estres_inicial', 5) <= 5:
        rec.append("Añade 1 sesión de HIIT en casa (20 min): alterna sprints en el sitio, jumping jacks y burpees con descansos cortos.")
    # Sueño
    if entrada.get('horas_sueno', 7) < 7:
        rec.append("Intenta dormir al menos 7 horas cada noche; la recuperación es clave para no abandonar.")
    # Dieta
    if 'tipo_dieta_cetogenica' in entrada and entrada['tipo_dieta_cetogenica'] == 1:
        rec.append("Mantén la dieta cetogénica, está siendo un gran aliado para tu objetivo.")
    elif 'tipo_dieta_balanceada' in entrada and entrada['tipo_dieta_balanceada'] == 1:
        rec.append("Podrías probar una dieta baja en carbohidratos; el modelo sugiere que ayudaría más que la balanceada.")
    # Autoestima / estrés
    if entrada.get('autoestima_inicial', 5) < 5:
        rec.append("Refuerza tu motivación: establece metas pequeñas y celebra cada logro.")
    if entrada.get('estres_inicial', 5) > 7:
        rec.append("Reduce el estrés con meditación, yoga suave o paseos al aire libre; mejorará tu adherencia al plan.")
    if not rec:
        rec.append("Mantén tus hábitos actuales, vas por muy buen camino.")
    return " | ".join(rec)

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json()
    df_input = pd.DataFrame([input_json])

    # Rellenar columnas faltantes con 0
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[features]

    proba = modelo.predict_proba(df_input)[:, 1][0]

    # SHAP values
    shap_vals = explainer.shap_values(df_input)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals[0]
    top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
    factores = {features[i]: shap_vals[i] for i in top_idx}

    # Recomendación
    recomendacion = generar_recomendacion(input_json, factores, target)

    return jsonify({
        'prob': proba,
        'factores': factores,
        'recomendacion': recomendacion,
        'model_version': version
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_version': version})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)