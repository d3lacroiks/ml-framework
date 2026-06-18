import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
import joblib
import json
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# 1. Cargar configuración
# -----------------------------------------------
if len(sys.argv) != 2:
    print("Uso: python train.py config/config_churn.json")
    sys.exit(1)

config_path = sys.argv[1]
with open(config_path, 'r') as f:
    config = json.load(f)

# -----------------------------------------------
# 2. Cargar datos
# -----------------------------------------------
df = pd.read_csv(config['dataset'])
print(f"Dataset cargado: {config['dataset']} ({len(df)} filas)")

# -----------------------------------------------
# 3. Limpieza básica
# -----------------------------------------------
if config['target_mapping']:
    df[config['target']] = df[config['target']].map(config['target_mapping'])

for col in config['numeric_columns']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
print(f"Filas después de limpieza: {len(df)}")

# -----------------------------------------------
# 4. Feature engineering
# -----------------------------------------------
if config['categorical_columns']:
    df = pd.get_dummies(df, columns=config['categorical_columns'], drop_first=True)

for feat_name, formula in config['engineered_features'].items():
    try:
        df[feat_name] = eval(formula, {"__builtins__": None}, {
            'df': df,
            'pd': pd,
            'np': np,
            **{col: df[col] for col in df.columns}
        })
    except Exception as e:
        print(f"No se pudo crear {feat_name}: {e}")

features = [c for c in df.columns if c not in [config['target']] + config['drop_columns']]
print(f"Features generadas: {len(features)}")

# -----------------------------------------------
# 5. Entrenamiento
# -----------------------------------------------
X = df[features]
y = df[config['target']]

tscv = TimeSeriesSplit(n_splits=3)
modelos = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    model = lgb.LGBMClassifier(
        scale_pos_weight=scale_pos,
        **config['model_params'],
        verbose=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, y_pred)
    print(f"  Fold {fold+1} AP: {ap:.4f}")
    modelos.append(model)

final_model = modelos[-1]

# -----------------------------------------------
# 6. Guardado del modelo
# -----------------------------------------------
os.makedirs('model_registry', exist_ok=True)
version = datetime.now().strftime('%Y%m%d_%H%M%S')
modelo_path = f'model_registry/model_{version}.pkl'
joblib.dump({
    'modelo': final_model,
    'features': features,
    'config': config
}, modelo_path)
print(f"✅ Modelo guardado en {modelo_path}")

with open('model_registry/active_version.txt', 'w') as f:
    f.write(version)
print(f"✅ Versión activa: {version}")