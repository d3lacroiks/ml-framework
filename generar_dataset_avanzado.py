import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# 1. Cargar dataset real de obesidad y extraer distribuciones empíricas
# -------------------------------------------------------------------
real = pd.read_csv('data/obesity_data.csv')
print(f"Dataset real cargado: {real.shape}")

age_mean, age_std = real['Age'].mean(), real['Age'].std()
height_mean, height_std = real['Height'].mean(), real['Height'].std()
weight_mean, weight_std = real['Weight'].mean(), real['Weight'].std()
faf_mean, faf_std = real['FAF'].mean(), real['FAF'].std()      # frecuencia de actividad física

# -------------------------------------------------------------------
# 2. Generar 5000 pacientes sintéticos con distribuciones empíricas
# -------------------------------------------------------------------
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

# Variables psicológicas con correlación no lineal al peso
df['autoestima_inicial'] = (8 - 0.02 * df['peso_inicial_kg'] + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)
# El estrés tiene una relación cuadrática suave con el peso (más estrés en obesidad severa)
df['estres_inicial'] = (2 + 0.01 * df['peso_inicial_kg']**2 / 100 + np.random.normal(0, 1.5, n)).clip(1, 10).astype(int)

# -------------------------------------------------------------------
# 3. Adherencia con factores contradictorios deliberados
# -------------------------------------------------------------------
# La adherencia depende de autoestima, estrés y ejercicio, pero añadimos
# un factor externo no observado (ruido fuerte + sesgo aleatorio)
adherencia_base = (
    20
    + 0.3 * df['autoestima_inicial']
    - 0.6 * df['estres_inicial']
    + 0.2 * df['horas_ejercicio_semana']
    + np.random.normal(0, 4, n)            # ruido grande para crear excepciones
)
# Añadimos un "factor externo" aleatorio que puede hacer que alguien con
# alta adherencia esperada realmente tenga baja adherencia (y viceversa)
factor_externo = np.random.choice([-8, 0, 8], n, p=[0.15, 0.70, 0.15])  # 15% de casos contradictorios
df['dias_adherencia'] = np.clip(adherencia_base + factor_externo, 1, 30).astype(int)

# -------------------------------------------------------------------
# 4. Pérdida de peso con interacciones no lineales y resistencia metabólica
# -------------------------------------------------------------------
# La pérdida de peso ya no es simplemente una suma lineal.
# Incluimos un término de interacción dieta-estrés, y una "resistencia metabólica"
# que hace que algunos pacientes pierdan poco aunque hagan todo bien.
perdida_ideal = (
    3.0
    + 1.2 * df['horas_ejercicio_semana']
    - 0.0015 * (df['calorias_diarias'] - 1800)
    + 0.6 * df['dias_adherencia'] / 10
    + np.where(df['tipo_dieta'] == 'cetogenica', 2.5, 0)
    + np.where(df['tipo_dieta'] == 'baja_carb', 1.2, 0)
)

# Interacción no lineal dieta-estrés (el estrés afecta más a quienes no hacen dieta cetogénica)
interaccion_dieta_estres = np.where(df['tipo_dieta'] != 'cetogenica',
                                    -0.2 * df['estres_inicial'],
                                    0)

# Resistencia metabólica aleatoria (15% de pacientes pierden menos de lo esperado)
resistencia = np.random.choice([0, -3], n, p=[0.85, 0.15])

# Ruido biológico
ruido = np.random.normal(0, 2.0, n)

df['perdida_kg'] = perdida_ideal + interaccion_dieta_estres + resistencia + ruido
df['perdida_kg'] = np.clip(df['perdida_kg'], -2, 25)  # puede subir ligeramente
df['peso_final_kg'] = df['peso_inicial_kg'] - df['perdida_kg']
df['peso_final_kg'] = df['peso_final_kg'].clip(45, 200)

# -------------------------------------------------------------------
# 5. Targets (éxito y abandono) con umbrales
# -------------------------------------------------------------------
df['exito_10pct'] = ((df['perdida_kg'] / df['peso_inicial_kg']) >= 0.10).astype(int)
df['abandono'] = (df['dias_adherencia'] < 21).astype(int)

# -------------------------------------------------------------------
# 6. Documentación de procedencia de variables
# -------------------------------------------------------------------
# Variables empíricas (distribuciones extraídas del dataset real): edad, altura, peso, actividad física
# Variables modeladas con reglas clínicas: autoestima, estrés, adherencia, pérdida de peso
# Se añadieron ruido gaussiano, interacciones no lineales y casos contradictorios deliberados.
# Ver README para más detalles.

print(f"Pacientes generados: {len(df)}")
print(f"Tasa de éxito (pérdida ≥10%): {df['exito_10pct'].mean():.2%}")
print(f"Tasa de abandono: {df['abandono'].mean():.2%}")

df.to_csv('data/programa_avanzado.csv', index=False)
print("Dataset guardado en data/programa_avanzado.csv")