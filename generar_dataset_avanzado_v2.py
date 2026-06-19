import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Cargar dataset real de obesidad y extraer distribuciones
# ------------------------------------------------------------
real = pd.read_csv('data/obesity_data.csv')
age_mean, age_std = real['Age'].mean(), real['Age'].std()
height_mean, height_std = real['Height'].mean(), real['Height'].std()
weight_mean, weight_std = real['Weight'].mean(), real['Weight'].std()
faf_mean, faf_std = real['FAF'].mean(), real['FAF'].std()

# ------------------------------------------------------------
# 2. Generar 5000 pacientes sintéticos
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
    # NUEVAS VARIABLES DE EJERCICIO DESGLOSADAS
    'horas_cardio_semana': np.random.normal(2, 1.5, n).clip(0, 10),
    'horas_fuerza_semana': np.random.normal(1, 1.2, n).clip(0, 8),
    'horas_hiit_semana': np.random.normal(0.5, 0.8, n).clip(0, 4),
    # ENTORNO Y EQUIPAMIENTO
    'tiene_pesas': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'tiene_bandas': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    'espacio': np.random.choice(['pequeno', 'mediano', 'grande'], n, p=[0.3, 0.5, 0.2]),
    # Variables anteriores
    'calorias_diarias': np.random.normal(2000, 400, n).clip(1000, 3500),
    'tipo_dieta': np.random.choice(['balanceada','baja_carb','cetogenica','mediterranea'], n, p=[0.4,0.3,0.2,0.1]),
    'horas_sueno': np.random.normal(7, 1.5, n).clip(4, 10),
    'autoestima_inicial': np.random.randint(1, 11, n),
    'estres_inicial': np.random.randint(1, 11, n)
})

# Autoestima y estrés correlacionados con peso
df['autoestima_inicial'] = (8 - 0.02 * df['peso_inicial_kg'] + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)
df['estres_inicial'] = (2 + 0.01 * df['peso_inicial_kg']**2 / 100 + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)

# ------------------------------------------------------------
# 3. Adherencia y pérdida de peso (similar al anterior)
# ------------------------------------------------------------
adherencia_base = (
    20 + 0.3 * df['autoestima_inicial'] - 0.6 * df['estres_inicial']
    + 0.2 * (df['horas_cardio_semana'] + df['horas_fuerza_semana'] + df['horas_hiit_semana'])
    + np.random.normal(0, 4, n)
)
df['dias_adherencia'] = np.clip(adherencia_base + np.random.choice([-8,0,8], n, p=[0.15,0.7,0.15]), 1, 30).astype(int)

# ------------------------------------------------------------
# 4. Pérdida de peso con tipos de ejercicio, resistencia metabólica y ruido
# ------------------------------------------------------------
# Resistencia metabólica aleatoria (15% de pacientes pierden menos de lo esperado)
resistencia_metabolica = np.random.choice([0, -3], n, p=[0.85, 0.15])

# Interacción dieta‑estrés
interaccion_dieta_estres = np.where(df['tipo_dieta'] != 'cetogenica',
                                    -0.2 * df['estres_inicial'],
                                    0)

perdida_ideal = (
    3.0
    + 1.0 * df['horas_cardio_semana']
    + 0.7 * df['horas_fuerza_semana']
    + 1.5 * df['horas_hiit_semana']
    - 0.0015 * (df['calorias_diarias'] - 1800)
    + 0.6 * df['dias_adherencia'] / 10
    + np.where(df['tipo_dieta'] == 'cetogenica', 2.5, 0)
    + np.where(df['tipo_dieta'] == 'baja_carb', 1.2, 0)
)

df['perdida_kg'] = perdida_ideal + interaccion_dieta_estres + resistencia_metabolica + np.random.normal(0, 2.0, n)
df['perdida_kg'] = np.clip(df['perdida_kg'], -2, 25)
df['peso_final_kg'] = df['peso_inicial_kg'] - df['perdida_kg']
df['peso_final_kg'] = df['peso_final_kg'].clip(45, 200)

# Targets
df['exito_10pct'] = ((df['perdida_kg'] / df['peso_inicial_kg']) >= 0.10).astype(int)
df['abandono'] = (df['dias_adherencia'] < 21).astype(int)

print(f"Pacientes generados: {len(df)}")
print(f"Tasa de éxito: {df['exito_10pct'].mean():.2%}")
print(f"Tasa de abandono: {df['abandono'].mean():.2%}")

df.to_csv('data/programa_avanzado_v2.csv', index=False)