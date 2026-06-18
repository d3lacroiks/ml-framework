import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Cargar dataset real de obesidad y obtener distribuciones
# ------------------------------------------------------------
real = pd.read_csv('data/obesity_data.csv')
print(f"Dataset real cargado: {real.shape}")

age_mean, age_std = real['Age'].mean(), real['Age'].std()
height_mean, height_std = real['Height'].mean(), real['Height'].std()
weight_mean, weight_std = real['Weight'].mean(), real['Weight'].std()
faf_mean, faf_std = real['FAF'].mean(), real['FAF'].std()      # frecuencia de actividad física (días/sem)

# ------------------------------------------------------------
# 2. Generar 5000 pacientes sintéticos con esas distribuciones
# ------------------------------------------------------------
np.random.seed(42)
n = 5000

df = pd.DataFrame({
    'id_paciente': range(1, n+1),
    'edad': np.random.normal(age_mean, age_std, n).clip(14, 80).astype(int),
    'sexo': np.random.choice(['M', 'F'], n),
    'peso_inicial_kg': np.random.normal(weight_mean, weight_std, n).clip(40, 200),
    'altura_cm': np.random.normal(height_mean, height_std, n).clip(1.3, 2.1) * 100,
    'actividad_fisica_semanal': np.random.normal(faf_mean, faf_std, n).clip(0, 7),
    'horas_ejercicio_semana': np.random.normal(faf_mean*2, faf_std, n).clip(0, 14),
    'calorias_diarias': np.random.normal(2000, 400, n).clip(1000, 3500),
    'tipo_dieta': np.random.choice(['balanceada','baja_carb','cetogenica','mediterranea'], n, p=[0.4,0.3,0.2,0.1]),
    'horas_sueno': np.random.normal(7, 1.5, n).clip(4, 10),
    'autoestima_inicial': np.random.randint(1, 11, n),   # se recalculará
    'estres_inicial': np.random.randint(1, 11, n)
})

# Correlación peso → autoestima/estrés (leve)
df['autoestima_inicial'] = (8 - 0.02 * df['peso_inicial_kg'] + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)
df['estres_inicial'] = (3 + 0.02 * df['peso_inicial_kg'] + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)

# ------------------------------------------------------------
# 3. Adherencia (días de seguimiento) → objetivo ~70% de pacientes con ≥21 días
# ------------------------------------------------------------
# Ajustamos la base para que la mayoría ronde los 25 días, pero con cola izquierda
adherencia_base = (
    22                                   # base un poco más baja
    + 0.25 * df['autoestima_inicial']
    - 0.7 * df['estres_inicial']         # el estrés pesa un poco más
    + 0.2 * df['horas_ejercicio_semana']
    + np.random.normal(0, 3, n)          # variabilidad
)
df['dias_adherencia'] = np.clip(adherencia_base, 1, 30).astype(int)

# ------------------------------------------------------------
# 4. Pérdida de peso (kg) → buscamos que ~65% pierda ≥10%
# ------------------------------------------------------------
# Aumentamos el efecto del ejercicio y la adherencia, pero bajamos ligeramente la base
perdida_base = (
    4.5                                          # kg base (más alta)
    + 1.5 * df['horas_ejercicio_semana']         # más efecto del ejercicio
    - 0.002 * (df['calorias_diarias'] - 1800)
    + 0.6 * df['dias_adherencia'] / 10
    + np.where(df['tipo_dieta'] == 'cetogenica', 2.5, 0)
    + np.where(df['tipo_dieta'] == 'baja_carb', 1.2, 0)
    + np.random.normal(0, 2.0, n)               # ruido individual
)
df['perdida_kg'] = np.clip(perdida_base, -2, 25)
df['peso_final_kg'] = df['peso_inicial_kg'] - df['perdida_kg']
df['peso_final_kg'] = df['peso_final_kg'].clip(45, 200)

# ------------------------------------------------------------
# 5. Variables objetivo
# ------------------------------------------------------------
df['exito_10pct'] = ((df['perdida_kg'] / df['peso_inicial_kg']) >= 0.10).astype(int)
df['abandono'] = (df['dias_adherencia'] < 21).astype(int)

print(f"Pacientes generados: {len(df)}")
print(f"Tasa de éxito (pérdida ≥10%): {df['exito_10pct'].mean():.2%}")
print(f"Tasa de abandono: {df['abandono'].mean():.2%}")

df.to_csv('data/programa_realista_exitoso.csv', index=False)
print("Dataset guardado en data/programa_realista_exitoso.csv")