import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Cargar dataset real de obesidad
# ------------------------------------------------------------
real = pd.read_csv('data/obesity_data.csv')
print(f"Dataset real cargado: {real.shape}")

# Vamos a extraer distribuciones de algunas columnas (ajusta nombres si tu CSV difiere)
# Supongamos que el dataset tiene: Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS, NObeyesdad
# Usaremos Age, Height, Weight, FAF (actividad física), CALC (consumo de alcohol), NCP (nº comidas), etc.

# Calcular parámetros para generación sintética
age_mean, age_std = real['Age'].mean(), real['Age'].std()
height_mean, height_std = real['Height'].mean(), real['Height'].std()
weight_mean, weight_std = real['Weight'].mean(), real['Weight'].std()
faf_mean, faf_std = real['FAF'].mean(), real['FAF'].std()   # frecuencia actividad física
ncp_mean, ncp_std = real['NCP'].mean(), real['NCP'].std()   # número de comidas principales
ch2o_mean, ch2o_std = real['CH2O'].mean(), real['CH2O'].std()  # consumo de agua

# ------------------------------------------------------------
# 2. Generar 5000 pacientes sintéticos basados en distribuciones reales
# ------------------------------------------------------------
np.random.seed(42)
n = 5000

df = pd.DataFrame({
    'id_paciente': range(1, n+1),
    'edad': np.random.normal(age_mean, age_std, n).clip(14, 80).astype(int),
    'sexo': np.random.choice(['M', 'F'], n),
    'peso_inicial_kg': np.random.normal(weight_mean, weight_std, n).clip(40, 200),
    'altura_cm': np.random.normal(height_mean, height_std, n).clip(1.3, 2.1) * 100,  # altura en cm (aprox 130-210)
    'actividad_fisica_semanal': np.random.normal(faf_mean, faf_std, n).clip(0, 7),  # días con actividad física
    'horas_ejercicio_semana': np.random.normal(faf_mean*2, faf_std, n).clip(0, 14),  # horas de ejercicio estimadas
    'calorias_diarias': np.random.normal(2000, 400, n).clip(1000, 3500),
    'tipo_dieta': np.random.choice(['balanceada','baja_carb','cetogenica','mediterranea'], n, p=[0.4,0.3,0.2,0.1]),
    'horas_sueno': np.random.normal(7, 1.5, n).clip(4, 10),
    'autoestima_inicial': np.random.randint(1, 11, n),  # lo crearemos con correlación después
    'estres_inicial': np.random.randint(1, 11, n)
})

# --- Dar realismo a autoestima y estrés (correlacionadas con peso y edad) ---
# Mayor peso -> menor autoestima, mayor estrés (simplificación)
df['autoestima_inicial'] = (8 - 0.02 * df['peso_inicial_kg'] + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)
df['estres_inicial'] = (3 + 0.02 * df['peso_inicial_kg'] + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)

# --- Simular adherencia (días que sigue el plan) ---
adherencia_base = (20 + 0.5 * df['autoestima_inicial'] - 0.5 * df['estres_inicial']
                   + 0.3 * df['horas_ejercicio_semana'] + np.random.normal(0, 4, n))
df['dias_adherencia'] = np.clip(adherencia_base, 1, 30).astype(int)

# --- Pérdida de peso (kg) ---
perdida_base = (
    4.0                                          # pérdida base más alta
    + 1.5 * df['horas_ejercicio_semana']         # mucho más efecto del ejercicio
    - 0.002 * (df['calorias_diarias'] - 1800)    # mismo efecto de calorías
    + 0.8 * df['dias_adherencia'] / 10           # más peso a la adherencia
    + np.where(df['tipo_dieta'] == 'cetogenica', 3.0, 0)  # bonus mayor
    + np.where(df['tipo_dieta'] == 'baja_carb', 1.5, 0)
    + np.random.normal(0, 1.5, n)                # menos ruido
)
df['perdida_kg'] = np.clip(perdida_base, -2, 25)
df['peso_final_kg'] = df['peso_inicial_kg'] - df['perdida_kg']
df['peso_final_kg'] = df['peso_final_kg'].clip(45, 200)

# --- Targets ---
df['exito_10pct'] = ((df['perdida_kg'] / df['peso_inicial_kg']) >= 0.10).astype(int)
df['abandono'] = (df['dias_adherencia'] < 21).astype(int)

print(f"Pacientes combinados generados: {len(df)}")
print(f"Tasa de éxito (pérdida ≥10%): {df['exito_10pct'].mean():.2%}")
print(f"Tasa de abandono: {df['abandono'].mean():.2%}")

df.to_csv('data/programa_realista.csv', index=False)