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
config = data['config']
print(f"Modelo cargado: {model_path}")

# Explainer SHAP
explainer = shap.TreeExplainer(modelo)

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

    return jsonify({
    'prob': proba,
    'factores': factores,
    'model_version': version
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_version': version})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)