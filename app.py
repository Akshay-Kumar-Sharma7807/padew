import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import time
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load model ────────────────────────────────────────────────────────────────
OUT = Path('output')
model_data   = joblib.load(OUT / 'padew_thesis_model.joblib')
base_pipe    = model_data['base_pipe']
rf_anom      = model_data['rf_anom']
threshold    = model_data['threshold']
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

# ── Cache SHAP explainers (built once, reused every rerun) ────────────────────
@st.cache_resource
def build_explainers():
    imputer  = base_pipe.named_steps['imputer']
    scaler   = base_pipe.named_steps['scaler']
    lr_model = base_pipe.named_steps['model']

    # Build a complete baseline row: DEFAULTS + binary features not in DEFAULTS
    baseline_full = {
        **DEFAULTS,
        'sex_male':   1,   # male baseline
        'ventilator': 1,
        'vasopressor': 1,
    }
    # reindex ensures every column in feature_cols is present, filling any gaps with 0
    bg        = pd.DataFrame([baseline_full]).reindex(columns=feature_cols, fill_value=0)
    bg_imp    = imputer.transform(bg)
    bg_scaled = scaler.transform(bg_imp)

    lr_expl = shap.LinearExplainer(lr_model, bg_scaled, feature_perturbation='interventional')
    rf_expl = shap.TreeExplainer(rf_anom)
    return lr_expl, rf_expl, imputer, scaler

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

# Vitals that show early warning signs — move throughout all 20 steps
_EARLY_VITALS = {'heart_rate', 'resp_rate', 'temperature', 'trend_hr', 'trend_rr', 'missing_rate'}
# Critical vitals — only start deteriorating from step 10 onward
_LATE_VITALS  = {'systolic_bp', 'diastolic_bp', 'map', 'spo2', 'lactate',
                  'creatinine', 'wbc', 'hemoglobin', 'fluid_balance_ml_24h',
                  'trend_map', 'trend_spo2'}

# ── Advance one simulation step ───────────────────────────────────────────────
def advance_simulation():
    step = st.session_state['sim_step']

    # Early vitals: spread linearly across all 20 steps
    frac_early = (step + 1) / TOTAL_STEPS

    # Late (critical) vitals: only start moving after step 9,
    # then accelerate over the remaining 10 steps
    frac_late  = max(0.0, (step - 9) / (TOTAL_STEPS - 10)) if TOTAL_STEPS > 10 else frac_early

    for key, target in DETERIORATION_TARGETS.items():
        start = DEFAULTS[key]
        frac  = frac_early if key in _EARLY_VITALS else frac_late
        new_val = start + (target - start) * frac
        lo, hi  = SLIDER_BOUNDS[key]
        if isinstance(DEFAULTS[key], float):
            st.session_state[key] = float(np.clip(new_val, lo, hi))
        else:
            st.session_state[key] = int(np.clip(round(new_val), lo, hi))

    # Activate ventilator halfway, vasopressor at 70%
    if frac_early >= 0.5:
        st.session_state['ventilator'] = True
    if frac_early >= 0.7:
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

# ── Big Alert Light ──────────────────────────────────────────────────────────
if alert:
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            background: #0e1117;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 240px;
            font-family: 'Segoe UI', sans-serif;
          }}
          .card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            background: rgba(180, 0, 0, 0.08);
            border: 1px solid rgba(255, 80, 80, 0.25);
            border-radius: 18px;
            padding: 28px 48px;
            backdrop-filter: blur(10px);
          }}
          .bulb {{
            width: 90px;
            height: 90px;
            border-radius: 50%;
            background: radial-gradient(circle at 38% 35%, #ff8080, #cc0000 55%, #7a0000);
            box-shadow:
              0 0 28px 8px rgba(220,0,0,0.85),
              0 0 70px 25px rgba(200,0,0,0.45);
            animation: pulse-red 0.9s ease-in-out infinite alternate;
            border: 3px solid rgba(255, 110, 110, 0.45);
          }}
          @keyframes pulse-red {{
            from {{ box-shadow: 0 0 28px 8px rgba(220,0,0,0.85), 0 0 70px 25px rgba(200,0,0,0.45); }}
            to   {{ box-shadow: 0 0 52px 20px rgba(255,0,0,1.0), 0 0 110px 45px rgba(220,0,0,0.75); }}
          }}
          .label {{
            color: #ff4444;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: 3px;
            text-shadow: 0 0 14px rgba(255, 50, 50, 0.9);
            animation: flicker 0.9s ease-in-out infinite alternate;
          }}
          @keyframes flicker {{
            from {{ opacity: 1; }}
            to   {{ opacity: 0.82; }}
          }}
          .score {{
            color: #ffaaaa;
            font-size: 0.88rem;
            letter-spacing: 0.5px;
          }}
          .mute-btn {{
            margin-top: 4px;
            padding: 8px 26px;
            background: rgba(200, 0, 0, 0.45);
            color: #fff;
            border: 1px solid rgba(255, 100, 100, 0.55);
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: background 0.2s;
          }}
          .mute-btn:hover {{ background: rgba(220, 30, 30, 0.7); }}
        </style>
        </head>
        <body>
        <div class="card">
          <div class="bulb"></div>
          <div class="label">⚠️ ALERT — HIGH RISK</div>
          <div class="score">Threshold: {threshold:.2f} &nbsp;|&nbsp; Score: {final_score:.1%}</div>
          <button class="mute-btn" id="alarmBtn" onclick="toggleAlarm()">🔇 Mute Alarm</button>
        </div>

        <script>
          var muted = false;
          var ctx = null;
          var alarmInterval = null;

          function getCtx() {{
            if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
            return ctx;
          }}

          function beep(freq, start, dur, vol) {{
            var c = getCtx();
            var osc = c.createOscillator();
            var g   = c.createGain();
            osc.connect(g); g.connect(c.destination);
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, start);
            g.gain.setValueAtTime(vol, start);
            g.gain.exponentialRampToValueAtTime(0.001, start + dur);
            osc.start(start); osc.stop(start + dur + 0.05);
          }}

          function siren() {{
            if (muted) return;
            var c = getCtx(), now = c.currentTime;
            beep(1040, now,        0.18, 0.5);
            beep(820,  now + 0.22, 0.18, 0.5);
            beep(1040, now + 0.46, 0.18, 0.5);
            beep(820,  now + 0.68, 0.18, 0.5);
          }}

          function toggleAlarm() {{
            muted = !muted;
            var btn = document.getElementById('alarmBtn');
            if (muted) {{
              btn.textContent = '🔔 Unmute Alarm';
              btn.style.background = 'rgba(70,70,70,0.5)';
            }} else {{
              btn.textContent = '🔇 Mute Alarm';
              btn.style.background = 'rgba(200,0,0,0.45)';
              siren();
            }}
          }}

          // Start alarm on first click (browser autoplay policy)
          document.addEventListener('click', function fire() {{
            siren();
            document.removeEventListener('click', fire);
          }});

          // Repeating siren every 1.1 s
          alarmInterval = setInterval(siren, 1100);
        </script>
        </body>
        </html>
        """,
        height=240,
    )
else:
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            background: #0e1117;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
            font-family: 'Segoe UI', sans-serif;
          }}
          .card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            background: rgba(0, 160, 60, 0.08);
            border: 1px solid rgba(0, 220, 100, 0.25);
            border-radius: 18px;
            padding: 28px 48px;
          }}
          .bulb {{
            width: 90px;
            height: 90px;
            border-radius: 50%;
            background: radial-gradient(circle at 38% 35%, #90ffb8, #00cc55 55%, #006622);
            box-shadow:
              0 0 28px 8px rgba(0,200,80,0.85),
              0 0 70px 25px rgba(0,180,60,0.45);
            animation: pulse-green 1.6s ease-in-out infinite alternate;
            border: 3px solid rgba(80, 255, 140, 0.45);
          }}
          @keyframes pulse-green {{
            from {{ box-shadow: 0 0 28px 8px rgba(0,200,80,0.85), 0 0 70px 25px rgba(0,180,60,0.45); }}
            to   {{ box-shadow: 0 0 46px 18px rgba(0,240,100,0.95), 0 0 100px 40px rgba(0,200,70,0.6); }}
          }}
          .label {{
            color: #00e060;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: 3px;
            text-shadow: 0 0 14px rgba(0, 220, 80, 0.9);
          }}
          .score {{
            color: #aaffcc;
            font-size: 0.88rem;
            letter-spacing: 0.5px;
          }}
        </style>
        </head>
        <body>
        <div class="card">
          <div class="bulb"></div>
          <div class="label">✅ STABLE — NO ALERT</div>
          <div class="score">Threshold: {threshold:.2f} &nbsp;|&nbsp; Score: {final_score:.1%}</div>
        </div>
        </body>
        </html>
        """,
        height=200,
    )


# ── SHAP Explanations ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🧠 SHAP Explanation")
st.caption("Why did the model give this score? SHAP breaks the prediction into individual feature contributions.")

try:
    lr_explainer, rf_explainer, _imp, _scaler = build_explainers()

    # Prepare scaled input for LR
    X_imp_s    = _imp.transform(X)
    X_scaled_s = _scaler.transform(X_imp_s)
    X_scaled_df = pd.DataFrame(X_scaled_s, columns=feature_cols)

    # ─ LR SHAP values (handle all SHAP version output shapes) ─
    lr_shap_raw = np.array(lr_explainer.shap_values(X_scaled_df))
    # Possible shapes: (n_feat,) | (1,n_feat) | (n_classes,1,n_feat) | (1,n_feat,n_classes)
    if lr_shap_raw.ndim == 1:
        lr_sv = lr_shap_raw                       # already 1-D
    elif lr_shap_raw.ndim == 2:
        lr_sv = lr_shap_raw[0]                    # (n_samples, n_feat) → first sample
    elif lr_shap_raw.ndim == 3 and lr_shap_raw.shape[0] == 2:
        lr_sv = lr_shap_raw[1, 0, :]              # (2, n_samples, n_feat) → class-1, sample-0
    elif lr_shap_raw.ndim == 3:
        lr_sv = lr_shap_raw[0, :, -1]             # (n_samples, n_feat, n_classes) → sample-0, class-1
    else:
        lr_sv = lr_shap_raw.flatten()[:len(feature_cols)]

    lr_ev_raw = lr_explainer.expected_value
    lr_ev = float(np.array(lr_ev_raw).flatten()[-1])

    lr_exp = shap.Explanation(
        values=lr_sv.astype(float),
        base_values=lr_ev,
        data=X_scaled_df.iloc[0].values,
        feature_names=feature_cols,
    )

    # ─ RF SHAP values ─
    rf_shap_raw = rf_explainer.shap_values(X)
    rf_sv_arr   = np.array(rf_shap_raw)
    if rf_sv_arr.ndim == 1:
        rf_sv = rf_sv_arr
    elif rf_sv_arr.ndim == 2:
        rf_sv = rf_sv_arr[0]                      # (n_samples, n_feat)
    elif rf_sv_arr.ndim == 3 and rf_sv_arr.shape[0] == 2:
        rf_sv = rf_sv_arr[1, 0, :]                # list-of-2 → class-1, sample-0
    elif rf_sv_arr.ndim == 3:
        rf_sv = rf_sv_arr[0, :, -1]               # (n_samples, n_feat, n_classes)
    else:
        rf_sv = rf_sv_arr.flatten()[:len(feature_cols)]

    rf_ev_raw = rf_explainer.expected_value
    rf_ev = float(np.array(rf_ev_raw).flatten()[-1])

    rf_exp = shap.Explanation(
        values=rf_sv.astype(float),
        base_values=rf_ev,
        data=X.iloc[0].values.astype(float),
        feature_names=feature_cols,
    )


    tab1, tab2 = st.tabs(["📉 Risk Model (Logistic Regression)", "🌲 Anomaly Model (Random Forest)"])

    # Dark matplotlib style matching Streamlit dark theme
    dark_style = {
        'axes.facecolor':   '#0e1117',
        'figure.facecolor': '#0e1117',
        'text.color':       '#fafafa',
        'axes.labelcolor':  '#fafafa',
        'xtick.color':      '#aaaaaa',
        'ytick.color':      '#fafafa',
        'axes.edgecolor':   '#333333',
        'grid.color':       '#222222',
    }

    with tab1:
        st.markdown("""
        **How to read:** Each bar shows how much a feature **pushed the score up ▶ (red)**
        or **pulled it down ◀ (blue)** from the baseline.  
        The sum of all bars + baseline = final predicted probability.
        """)
        with plt.rc_context(dark_style):
            shap.plots.waterfall(lr_exp, max_display=12, show=False)
            fig1 = plt.gcf()
            fig1.patch.set_facecolor('#0e1117')
            for ax in fig1.axes:
                ax.set_facecolor('#0e1117')
                ax.tick_params(colors='#cccccc')
            fig1.set_size_inches(9, 5)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

    with tab2:
        st.markdown("""
        **How to read:** Shows which vitals made the **Random Forest flag this patient as anomalous**.
        Longer red bars = stronger push toward anomaly; blue bars = toward normal.
        """)
        rf_df = pd.DataFrame({'feature': feature_cols, 'shap': rf_sv})
        rf_df = rf_df.iloc[rf_df['shap'].abs().argsort()]
        colors = ['#e05252' if v > 0 else '#5285e0' for v in rf_df['shap']]

        with plt.rc_context(dark_style):
            fig2, ax2 = plt.subplots(figsize=(9, 6))
            ax2.barh(rf_df['feature'], rf_df['shap'], color=colors, edgecolor='none', height=0.65)
            ax2.axvline(0, color='#555555', linewidth=0.8)
            ax2.set_xlabel('SHAP value (impact on anomaly score)', color='#aaaaaa', fontsize=9)
            ax2.set_title('Random Forest Anomaly — Feature Impact', color='#fafafa', fontsize=11, pad=10)
            ax2.tick_params(axis='y', labelsize=8.5)
            ax2.tick_params(axis='x', labelsize=8)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

except Exception as e:
    st.warning(f"⚠️ SHAP explanation unavailable: {e}")
    # Fallback: simple coefficient table
    coef    = base_pipe.named_steps['model'].coef_[0]
    scaler  = base_pipe.named_steps['scaler']
    imputer = base_pipe.named_steps['imputer']
    X_imp_f       = imputer.transform(X)
    X_scaled_f    = scaler.transform(X_imp_f)
    contributions = X_scaled_f[0] * coef
    contrib_df = pd.DataFrame({'feature': feature_cols, 'contribution': contributions})
    contrib_df = contrib_df.iloc[contrib_df['contribution'].abs().argsort()[::-1]]
    st.dataframe(contrib_df.head(10).style.format({'contribution': '{:.3f}'}), use_container_width=True)

st.divider()
st.caption("Model: Logistic Regression + Random Forest ensemble | Threshold tuned for clinical utility")

# ── Auto-rerun while simulation is active ─────────────────────────────────────
if st.session_state['simulating']:
    time.sleep(STEP_DELAY)
    st.rerun()
