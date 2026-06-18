import pandas as pd
import numpy as np

np.random.seed(42)
n_pacientes = 3000

data = {
    'id_paciente': range(1, n_pacientes+1),
    'edad': np.random.randint(18, 70, n_pacientes),
    'sexo': np.random.choice(['M', 'F'], n_pacientes),
    'peso_inicial_kg': np.random.normal(85, 15, n_pacientes).clip(50, 150),
    'altura_cm': np.random.normal(170, 10, n_pacientes).clip(140, 200),
    'horas_ejercicio_semana': np.random.randint(0, 14, n_pacientes),  # hasta 14 horas
    'calorias_diarias': np.random.normal(1800, 400, n_pacientes).clip(1000, 3000),
    'dias_seguimiento': np.random.randint(28, 31, n_pacientes),
}

df = pd.DataFrame(data)

# Nueva fórmula de pérdida de peso (más realista)
# Pérdida base de 2 kg + efecto del ejercicio y déficit calórico
df['perdida_kg'] = (
    2.0
    + 1.2 * df['horas_ejercicio_semana'] * df['dias_seguimiento']/7
    + 0.005 * (2000 - df['calorias_diarias']) * df['dias_seguimiento']/30
    + np.random.normal(0, 2.0, n_pacientes)  # variabilidad individual
)
df['peso_final_kg'] = df['peso_inicial_kg'] - df['perdida_kg']
df['peso_final_kg'] = df['peso_final_kg'].clip(45, 180)

# Target: ¿perdió al menos el 10% de su peso inicial?
df['perdida_10pct'] = (df['perdida_kg'] / df['peso_inicial_kg'] >= 0.10).astype(int)

print(f"Pacientes generados: {len(df)}")
print(f"Tasa de éxito (pérdida ≥10%): {df['perdida_10pct'].mean():.2%}")

df.to_csv('data/pacientes.csv', index=False)