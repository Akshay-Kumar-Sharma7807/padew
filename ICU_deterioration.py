import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

OUT = Path('output')
OUT.mkdir(exist_ok=True)

# Load MIMIC-IV data (or sample data if MIMIC not available)
csv_path = OUT / 'mimic_icu_dataset.csv'

# Check if MIMIC data exists, otherwise create sample
if not csv_path.exists():
    print("MIMIC-IV data not found. Creating sample data...")
    from load_mimic_data import create_sample_mimic_format
    df = create_sample_mimic_format()
    df.to_csv(csv_path, index=False)
    print(f"Created sample data: {len(df)} rows at {csv_path}")
else:
    print(f"Loading MIMIC-IV data from {csv_path}")
    df = pd.read_csv(csv_path)

# Features aligned with thesis chapter 3
feature_cols = [
    'age','sex_male','charlson_index','heart_rate','systolic_bp','diastolic_bp','map','spo2',
    'resp_rate','temperature','lactate','creatinine','wbc','hemoglobin','potassium','sodium',
    'ventilator','vasopressor','fluid_balance_ml_24h','trend_hr','trend_rr','trend_map','trend_spo2','missing_rate'
]
target_col = 'deterioration'
lead_col = 'lead_time_hours'

X = df[feature_cols].copy()
y = df[target_col].astype(int).copy()
lead = df[lead_col].copy()

# Split with stratification for stable evaluation
X_train, X_test, y_train, y_test, lead_train, lead_test = train_test_split(
    X, y, lead, test_size=0.2, random_state=42, stratify=y
)

# Main PADEW training model: interpretable calibrated classifier on top of anomaly-like signals
base_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42))
])
base_pipe.fit(X_train, y_train)
prob_test = base_pipe.predict_proba(X_test)[:, 1]
prob_train = base_pipe.predict_proba(X_train)[:, 1]

# Secondary anomaly detector for personalization signal
rf_anom = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=5,
    class_weight='balanced_subsample',
    random_state=42
)
rf_anom.fit(X_train, y_train)
anom_test = rf_anom.predict_proba(X_test)[:, 1]
anom_train = rf_anom.predict_proba(X_train)[:, 1]

# Combine anomaly and probability for final alert score
final_train_score = 0.6 * prob_train + 0.4 * anom_train
final_test_score = 0.6 * prob_test + 0.4 * anom_test

# Threshold tuned to maximize recall while keeping reasonable precision
candidate_thr = np.linspace(0.1, 0.9, 81)
rows = []
for t in candidate_thr:
    pred = (final_train_score >= t).astype(int)
    if pred.sum() == 0:
        continue
    p = precision_score(y_train, pred, zero_division=0)
    r = recall_score(y_train, pred, zero_division=0)
    f1 = f1_score(y_train, pred, zero_division=0)
    lead_ok = ((lead_train[y_train == 1] >= 4).mean() if (y_train == 1).any() else 0)
    score = 0.45 * f1 + 0.35 * r + 0.20 * p + 0.05 * lead_ok
    rows.append((t, p, r, f1, score))

tab = pd.DataFrame(rows, columns=['thr','precision','recall','f1','score'])
best_thr = float(tab.sort_values('score', ascending=False).iloc[0]['thr'])

# Final predictions
pred_test = (final_test_score >= best_thr).astype(int)

# Metrics
auroc = roc_auc_score(y_test, final_test_score)
auprc = average_precision_score(y_test, final_test_score)
acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test, zero_division=0)
rec = recall_score(y_test, pred_test, zero_division=0)
f1 = f1_score(y_test, pred_test, zero_division=0)
cm = confusion_matrix(y_test, pred_test)

tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp) if (tn + fp) else np.nan

# Lead-time evaluation
lead_mask = (y_test == 1)
lead_values = lead_test[lead_mask].dropna().values
lead_mean = float(np.mean(lead_values)) if len(lead_values) else np.nan
lead_median = float(np.median(lead_values)) if len(lead_values) else np.nan
lead_ge_4 = float((lead_values >= 4).mean()) if len(lead_values) else np.nan

# Save outputs
pred_df = X_test.copy()
pred_df['true_label'] = y_test.values
pred_df['lead_time_hours'] = lead_test.values
pred_df['probability'] = prob_test
pred_df['anom_score'] = anom_test
pred_df['final_score'] = final_test_score
pred_df['alert'] = pred_test
pred_df.to_csv(OUT / 'thesis_test_predictions.csv', index=False)

metrics = pd.DataFrame([{
    'threshold': best_thr,
    'auroc': auroc,
    'auprc': auprc,
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1': f1,
    'specificity': spec,
    'tn': tn,
    'fp': fp,
    'fn': fn,
    'tp': tp,
    'mean_lead_time_hours': lead_mean,
    'median_lead_time_hours': lead_median,
    'proportion_lead_ge_4h': lead_ge_4,
    'train_n': len(X_train),
    'test_n': len(X_test),
    'n_features': len(feature_cols)
}])
metrics.to_csv(OUT / 'thesis_metrics.csv', index=False)

# Explanation outputs using logistic regression coefficients and SHAP
imputer = base_pipe.named_steps['imputer']
scaler = base_pipe.named_steps['scaler']
model = base_pipe.named_steps['model']
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
X_train_scaled = scaler.transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

coef = pd.DataFrame({'feature': feature_cols, 'coefficient': model.coef_[0]})
coef['abs_coef'] = coef['coefficient'].abs()
coef = coef.sort_values('abs_coef', ascending=False).drop(columns='abs_coef')
coef.to_csv(OUT / 'thesis_logistic_coefficients.csv', index=False)

# SHAP for clinician-facing explanation on subset
explainer = shap.LinearExplainer(model, X_train_scaled, feature_names=feature_cols)
shap_values = explainer(X_test_scaled)

# Global importance
shap_mean = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': shap_mean}).sort_values('mean_abs_shap', ascending=False)
shap_df.to_csv(OUT / 'thesis_shap_importance.csv', index=False)

# Save a small per-patient explanation for top 10 alerts
alert_idx = np.where(pred_test == 1)[0]
rows = []
for i in alert_idx[:10]:
    vals = shap_values.values[i]
    top = np.argsort(np.abs(vals))[::-1][:5]
    for j in top:
        rows.append({
            'row_index': int(i),
            'feature': feature_cols[j],
            'shap_value': float(vals[j]),
            'feature_value': float(X_test.iloc[i][feature_cols[j]])
        })
exp_df = pd.DataFrame(rows)
exp_df.to_csv(OUT / 'thesis_top_alert_explanations.csv', index=False)

# Model artifact
joblib.dump({'base_pipe': base_pipe, 'rf_anom': rf_anom, 'threshold': best_thr, 'feature_cols': feature_cols}, OUT / 'padew_thesis_model.joblib')

# Plots
sns.set(style='whitegrid')
plt.figure(figsize=(7,5))
sns.histplot(final_test_score[y_test==0], color='steelblue', label='No deterioration', kde=True, stat='density', bins=25)
sns.histplot(final_test_score[y_test==1], color='darkred', label='Deterioration', kde=True, stat='density', bins=25, alpha=0.55)
plt.axvline(best_thr, color='black', linestyle='--', label='Threshold')
plt.title('PADEW Final Score Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'padew_score_distribution.png', dpi=200)
plt.close()

plt.figure(figsize=(7,5))
plot_df = shap_df.head(12).iloc[::-1]
plt.barh(plot_df['feature'], plot_df['mean_abs_shap'], color='teal')
plt.title('Global SHAP Feature Importance')
plt.tight_layout()
plt.savefig(OUT / 'padew_shap_importance.png', dpi=200)
plt.close()

print(metrics.to_string(index=False))
print('\nSaved files in output/: thesis_metrics.csv, thesis_test_predictions.csv, thesis_logistic_coefficients.csv, thesis_shap_importance.csv, thesis_top_alert_explanations.csv, padew_thesis_model.joblib, padew_score_distribution.png, padew_shap_importance.png')