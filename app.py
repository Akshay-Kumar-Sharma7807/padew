import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path

# ── Load model ────────────────────────────────────────────────────────────────
OUT = Path('output')
model_data = joblib.load(OUT / 'padew_thesis_model.joblib')
base_pipe   = model_data['base_pipe']
rf_anom     = model_data['rf_anom']
threshold   = model_data['threshold']
feature_cols = model_data['feature_cols']

st.set_page_config(page_title="ICU Deterioration Predictor", page_icon="🏥", layout="wide")

# ── Default (healthy baseline) values ─────────────────────────────────────────
DEFAULTS = {
    'age': 65,
    'charlson_index': 3,
    'heart_rate': 80,
    'systolic_bp': 130,
    'diastolic_bp': 75,
    'map': 90,
    'spo2': 97,
    'resp_rate': 16,
    'temperature': 37.0,
    'lactate': 1.5,
    'creatinine': 0.9,
    'wbc': 8,
    'hemoglobin': 12,
    'potassium': 4.2,
    'sodium': 140,
    'fluid_balance_ml_24h': 500,
    'trend_hr': 0,
    'trend_rr': 0,
    'trend_map': 0,
    'trend_spo2': 0,
    'missing_rate': 0.1,
}

# Target (deteriorating patient) values — reached step-by-step during simulation
DETERIORATION_TARGETS = {
    'heart_rate': 138,
    'systolic_bp': 78,
    'diastolic_bp': 48,
    'map': 55,
    'spo2': 84,
    'resp_rate': 36,
    'temperature': 39.2,
    'lactate': 9.5,
    'creatinine': 4.5,
    'wbc': 22,
    'hemoglobin': 7,
    'fluid_balance_ml_24h': -1500,
    'trend_hr': 22,
    'trend_rr': 14,
    'trend_map': -25,
    'trend_spo2': -8,
    'missing_rate': 0.45,
}

TOTAL_STEPS = 20   # number of animation frames
STEP_DELAY  = 0.6  # seconds between frames

SLIDER_BOUNDS = {
    'heart_rate':           (50,    180),
    'systolic_bp':          (70,    200),
    'diastolic_bp':         (40,    120),
    'map':                  (50,    140),
    'spo2':                 (80,    100),
    'resp_rate':            (8,     40),
    'temperature':          (35.0,  40.0),
    'lactate':              (0.5,   15.0),
    'creatinine':           (0.4,   8.0),
    'wbc':                  (2,     30),
    'hemoglobin':           (5,     18),
    'potassium':            (2.5,   6.5),
    'sodium':               (125,   155),
    'fluid_balance_ml_24h': (-2000, 4000),
    'trend_hr':             (-20,   30),
    'trend_rr':             (-10,   20),
    'trend_map':            (-30,   20),
    'trend_spo2':           (-10,   5),
    'missing_rate':         (0.0,   0.6),
    'age':                  (18,    95),
    'charlson_index':       (0,     15),
}

# ── Session-state initialisation ───────────────────────────────────────────────
if 'initialized' not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state['sex_male']    = True
    st.session_state['ventilator']  = False
    st.session_state['vasopressor'] = False
    st.session_state['sim_step']    = 0
    st.session_state['simulating']  = False
    st.session_state['initialized'] = True

# ── Advance one simulation step ───────────────────────────────────────────────
def advance_simulation():
    step = st.session_state['sim_step']
    frac = (step + 1) / TOTAL_STEPS

    for key, target in DETERIORATION_TARGETS.items():
        start   = DEFAULTS[key]
        new_val = start + (target - start) * frac
        lo, hi  = SLIDER_BOUNDS[key]
        if isinstance(DEFAULTS[key], float):
            st.session_state[key] = float(np.clip(new_val, lo, hi))
        else:
            st.session_state[key] = int(np.clip(round(new_val), lo, hi))

    # Activate ventilator halfway, vasopressor at 70%
    if frac >= 0.5:
        st.session_state['ventilator'] = True
    if frac >= 0.7:
        st.session_state['vasopressor'] = True

    st.session_state['sim_step'] += 1
    if st.session_state['sim_step'] >= TOTAL_STEPS:
        st.session_state['simulating'] = False

# Advance before rendering if actively simulating
if st.session_state['simulating']:
    advance_simulation()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 ICU Patient Deterioration Prediction")
st.markdown("### PADEW Early Warning System")

# ── Simulate / Reset buttons ──────────────────────────────────────────────────
btn_col1, btn_col2, info_col = st.columns([1, 1, 6])

with btn_col1:
    sim_label = "⏸ Pause" if st.session_state['simulating'] else "▶ Simulate"
    if st.button(sim_label, use_container_width=True, type="primary"):
        if st.session_state['simulating']:
            # Pause
            st.session_state['simulating'] = False
        else:
            # Start fresh from healthy baseline
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state['ventilator']  = False
            st.session_state['vasopressor'] = False
            st.session_state['sim_step']    = 0
            st.session_state['simulating']  = True
        st.rerun()

with btn_col2:
    if st.button("🔄 Reset", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state['sex_male']    = True
        st.session_state['ventilator']  = False
        st.session_state['vasopressor'] = False
        st.session_state['sim_step']    = 0
        st.session_state['simulating']  = False
        st.rerun()

with info_col:
    if st.session_state['simulating']:
        st.info("🔴 Simulation running — watch the sliders move as the patient deteriorates…")
    elif st.session_state['sim_step'] == TOTAL_STEPS:
        st.warning("⚠️ Simulation complete. Patient has reached critical deterioration state.")

# Simulation progress bar
if st.session_state['sim_step'] > 0:
    st.progress(
        st.session_state['sim_step'] / TOTAL_STEPS,
        text=f"Simulation: step {st.session_state['sim_step']}/{TOTAL_STEPS}"
    )

# ── Sidebar sliders (all driven by session_state) ─────────────────────────────
st.sidebar.header("Patient Features")

def get_user_input():
    return {
        'age': st.sidebar.slider(
            'Age', 18, 95, value=st.session_state['age'], key='age'),
        'sex_male': st.sidebar.radio(
            'Sex', ['Female', 'Male'],
            index=1 if st.session_state['sex_male'] else 0,
            key='sex_male_radio') == 'Male',
        'charlson_index': st.sidebar.slider(
            'Charlson Index', 0, 15, value=st.session_state['charlson_index'], key='charlson_index'),
        'heart_rate': st.sidebar.slider(
            'Heart Rate (bpm)', 50, 180, value=st.session_state['heart_rate'], key='heart_rate'),
        'systolic_bp': st.sidebar.slider(
            'Systolic BP (mmHg)', 70, 200, value=st.session_state['systolic_bp'], key='systolic_bp'),
        'diastolic_bp': st.sidebar.slider(
            'Diastolic BP (mmHg)', 40, 120, value=st.session_state['diastolic_bp'], key='diastolic_bp'),
        'map': st.sidebar.slider(
            'Mean Arterial Pressure', 50, 140, value=st.session_state['map'], key='map'),
        'spo2': st.sidebar.slider(
            'SpO2 (%)', 80, 100, value=st.session_state['spo2'], key='spo2'),
        'resp_rate': st.sidebar.slider(
            'Respiratory Rate', 8, 40, value=st.session_state['resp_rate'], key='resp_rate'),
        'temperature': st.sidebar.slider(
            'Temperature (°C)', 35.0, 40.0, value=float(st.session_state['temperature']), key='temperature'),
        'lactate': st.sidebar.slider(
            'Lactate (mmol/L)', 0.5, 15.0, value=float(st.session_state['lactate']), key='lactate'),
        'creatinine': st.sidebar.slider(
            'Creatinine (mg/dL)', 0.4, 8.0, value=float(st.session_state['creatinine']), key='creatinine'),
        'wbc': st.sidebar.slider(
            'WBC (x10³/µL)', 2, 30, value=st.session_state['wbc'], key='wbc'),
        'hemoglobin': st.sidebar.slider(
            'Hemoglobin (g/dL)', 5, 18, value=st.session_state['hemoglobin'], key='hemoglobin'),
        'potassium': st.sidebar.slider(
            'Potassium (mEq/L)', 2.5, 6.5, value=float(st.session_state['potassium']), key='potassium'),
        'sodium': st.sidebar.slider(
            'Sodium (mEq/L)', 125, 155, value=st.session_state['sodium'], key='sodium'),
        'ventilator': st.sidebar.checkbox(
            'On Ventilator', value=st.session_state['ventilator'], key='ventilator'),
        'vasopressor': st.sidebar.checkbox(
            'On Vasopressor', value=st.session_state['vasopressor'], key='vasopressor'),
        'fluid_balance_ml_24h': st.sidebar.slider(
            'Fluid Balance (mL/24h)', -2000, 4000,
            value=st.session_state['fluid_balance_ml_24h'], key='fluid_balance_ml_24h'),
        'trend_hr': st.sidebar.slider(
            'HR Trend', -20, 30, value=st.session_state['trend_hr'], key='trend_hr'),
        'trend_rr': st.sidebar.slider(
            'RR Trend', -10, 20, value=st.session_state['trend_rr'], key='trend_rr'),
        'trend_map': st.sidebar.slider(
            'MAP Trend', -30, 20, value=st.session_state['trend_map'], key='trend_map'),
        'trend_spo2': st.sidebar.slider(
            'SpO2 Trend', -10, 5, value=st.session_state['trend_spo2'], key='trend_spo2'),
        'missing_rate': st.sidebar.slider(
            'Missing Data Rate', 0.0, 0.6,
            value=float(st.session_state['missing_rate']), key='missing_rate'),
    }

input_data = get_user_input()

# ── Prediction ────────────────────────────────────────────────────────────────
X = pd.DataFrame([input_data])[feature_cols]

prob        = base_pipe.predict_proba(X)[0, 1]
anom        = rf_anom.predict_proba(X)[0, 1]
final_score = 0.6 * prob + 0.4 * anom
alert       = final_score >= threshold

# ── Results ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Prediction Results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Logistic Regression Prob", f"{prob:.1%}")
with col2:
    st.metric("Anomaly Score", f"{anom:.1%}")
with col3:
    st.metric("Final Score", f"{final_score:.1%}")

if alert:
    st.error(f"⚠️ ALERT: High risk of deterioration! (Threshold: {threshold:.2f})")
else:
    st.success(f"✅ No alert (Threshold: {threshold:.2f})")

# ── Feature Contributions ─────────────────────────────────────────────────────
st.divider()
st.subheader("Feature Contributions")

coef    = base_pipe.named_steps['model'].coef_[0]
scaler  = base_pipe.named_steps['scaler']
imputer = base_pipe.named_steps['imputer']

X_imp         = imputer.transform(X)
X_scaled      = scaler.transform(X_imp)
contributions = X_scaled[0] * coef

contrib_df = pd.DataFrame({
    'feature':      feature_cols,
    'contribution': contributions,
}).sort_values('contribution', key=abs, ascending=False)

st.dataframe(
    contrib_df.head(10).style.format({'contribution': '{:.3f}'}),
    use_container_width=True,
)

st.divider()
st.caption("Model: Logistic Regression + Random Forest ensemble | Threshold tuned for clinical utility")

# ── Auto-rerun while simulation is active ─────────────────────────────────────
if st.session_state['simulating']:
    time.sleep(STEP_DELAY)
    st.rerun()