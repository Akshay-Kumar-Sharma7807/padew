import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n_samples = 2000
n_deterioration = int(n_samples * 0.15)  # 15% deterioration rate

# Generate non-deterioration patients
n_normal = n_samples - n_deterioration

# Base features for normal patients
data = {
    'age': np.random.normal(65, 15, n_samples).clip(18, 95),
    'sex_male': np.random.binomial(1, 0.55, n_samples),
    'charlson_index': np.random.poisson(3, n_samples).clip(0, 15),
    'heart_rate': np.random.normal(80, 12, n_samples).clip(50, 140),
    'systolic_bp': np.random.normal(130, 18, n_samples).clip(90, 180),
    'diastolic_bp': np.random.normal(75, 10, n_samples).clip(50, 100),
    'map': np.random.normal(90, 12, n_samples).clip(60, 120),
    'spo2': np.random.normal(97, 2, n_samples).clip(90, 100),
    'resp_rate': np.random.normal(16, 3, n_samples).clip(10, 28),
    'temperature': np.random.normal(37.0, 0.5, n_samples).clip(36.0, 38.5),
    'lactate': np.random.exponential(1.5, n_samples).clip(0.5, 8),
    'creatinine': np.random.exponential(0.9, n_samples).clip(0.5, 4),
    'wbc': np.random.normal(8, 3, n_samples).clip(3, 20),
    'hemoglobin': np.random.normal(12, 2, n_samples).clip(7, 17),
    'potassium': np.random.normal(4.2, 0.4, n_samples).clip(3.0, 5.5),
    'sodium': np.random.normal(140, 3, n_samples).clip(130, 150),
    'ventilator': np.random.binomial(1, 0.1, n_samples),
    'vasopressor': np.random.binomial(1, 0.05, n_samples),
    'fluid_balance_ml_24h': np.random.normal(500, 800, n_samples).clip(-1000, 2500),
    'trend_hr': np.random.normal(0, 3, n_samples),
    'trend_rr': np.random.normal(0, 2, n_samples),
    'trend_map': np.random.normal(0, 5, n_samples),
    'trend_spo2': np.random.normal(0, 1, n_samples),
    'missing_rate': np.random.uniform(0, 0.3, n_samples),
}

# Add deterioration signals to first n_deterioration patients
deterioration_indices = np.arange(n_deterioration)

# Increase heart rate
data['heart_rate'][deterioration_indices] += np.random.normal(15, 5, n_deterioration)
# Decrease BP
data['systolic_bp'][deterioration_indices] -= np.random.normal(20, 8, n_deterioration)
data['diastolic_bp'][deterioration_indices] -= np.random.normal(10, 5, n_deterioration)
data['map'][deterioration_indices] -= np.random.normal(15, 6, n_deterioration)
# Decrease O2
data['spo2'][deterioration_indices] -= np.random.normal(4, 2, n_deterioration)
# Increase resp rate
data['resp_rate'][deterioration_indices] += np.random.normal(6, 3, n_deterioration)
# Increase lactate
data['lactate'][deterioration_indices] += np.random.normal(3, 1.5, n_deterioration)
# Increase creatinine
data['creatinine'][deterioration_indices] += np.random.normal(0.8, 0.4, n_deterioration)
# Increase WBC
data['wbc'][deterioration_indices] += np.random.normal(4, 2, n_deterioration)
# Decrease hemoglobin
data['hemoglobin'][deterioration_indices] -= np.random.normal(1.5, 0.8, n_deterioration)
# More ventilator/vasopressor
data['ventilator'][deterioration_indices] = np.random.binomial(1, 0.4, n_deterioration)
data['vasopressor'][deterioration_indices] = np.random.binomial(1, 0.3, n_deterioration)
# Negative fluid balance more common
data['fluid_balance_ml_24h'][deterioration_indices] -= np.random.normal(500, 300, n_deterioration)
# Trends worsen
data['trend_hr'][deterioration_indices] += np.random.normal(8, 3, n_deterioration)
data['trend_rr'][deterioration_indices] += np.random.normal(4, 2, n_deterioration)
data['trend_map'][deterioration_indices] -= np.random.normal(8, 4, n_deterioration)
data['trend_spo2'][deterioration_indices] -= np.random.normal(2, 1, n_deterioration)
# More missing data
data['missing_rate'][deterioration_indices] += np.random.uniform(0.1, 0.2, n_deterioration)

# Clip values to realistic ranges
data['heart_rate'] = data['heart_rate'].clip(50, 180)
data['systolic_bp'] = data['systolic_bp'].clip(70, 200)
data['diastolic_bp'] = data['diastolic_bp'].clip(40, 120)
data['map'] = data['map'].clip(50, 140)
data['spo2'] = data['spo2'].clip(80, 100)
data['resp_rate'] = data['resp_rate'].clip(8, 40)
data['temperature'] = data['temperature'].clip(35.0, 40.0)
data['lactate'] = data['lactate'].clip(0.5, 15)
data['creatinine'] = data['creatinine'].clip(0.4, 8)
data['wbc'] = data['wbc'].clip(2, 30)
data['hemoglobin'] = data['hemoglobin'].clip(5, 18)
data['potassium'] = data['potassium'].clip(2.5, 6.5)
data['sodium'] = data['sodium'].clip(125, 155)
data['fluid_balance_ml_24h'] = data['fluid_balance_ml_24h'].clip(-2000, 4000)
data['trend_hr'] = data['trend_hr'].clip(-20, 30)
data['trend_rr'] = data['trend_rr'].clip(-10, 20)
data['trend_map'] = data['trend_map'].clip(-30, 20)
data['trend_spo2'] = data['trend_spo2'].clip(-10, 5)
data['missing_rate'] = data['missing_rate'].clip(0, 0.6)

# Create target and lead time
deterioration = np.zeros(n_samples, dtype=int)
deterioration[deterioration_indices] = 1

# Lead time: 1-12 hours for deterioration patients
lead_time_hours = np.zeros(n_samples)
lead_time_hours[deterioration_indices] = np.random.uniform(1, 12, n_deterioration)

df = pd.DataFrame(data)
df['deterioration'] = deterioration
df['lead_time_hours'] = lead_time_hours

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
OUT = Path('output')
OUT.mkdir(exist_ok=True)
df.to_csv(OUT / 'synthetic_icu_padew_dataset.csv', index=False)

print(f"Generated {n_samples} samples")
print(f"  - Deterioration: {n_deterioration} ({100*n_deterioration/n_samples:.1f}%)")
print(f"  - No deterioration: {n_normal} ({100*n_normal/n_samples:.1f}%)")
print(f"Saved to {OUT / 'synthetic_icu_padew_dataset.csv'}")