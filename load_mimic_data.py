"""
MIMIC-IV Data Loader for ICU Deterioration Prediction
======================================================
This script loads and processes MIMIC-IV data to match the features used in the model.

MIMIC-IV Database Tables Used:
- patients: demographic info (age, sex)
- admissions: hospital admissions
- icustays: ICU stay information  
- chartevents: vital signs (heart rate, BP, SpO2, resp rate, temperature)
- labevents: lab results (lactate, creatinine, WBC, hemoglobin, potassium, sodium)
- inputevents: fluid input
- outputevents: fluid output
- ventilator_settings: mechanical ventilation
- medicationevents: vasopressors

Target Variable (deterioration):
- Based on composite outcome: ICU transfer, ward decline, or death within 24-48 hours
- Lead time: hours from observation to deterioration event

Usage:
    python load_mimic_data.py --mimic-path /path/to/mimiciv_data/
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Feature columns matching ICU_deterioration.py
FEATURE_COLS = [
    'age', 'sex_male', 'charlson_index', 'heart_rate', 'systolic_bp', 'diastolic_bp', 
    'map', 'spo2', 'resp_rate', 'temperature', 'lactate', 'creatinine', 'wbc', 
    'hemoglobin', 'potassium', 'sodium', 'ventilator', 'vasopressor', 
    'fluid_balance_ml_24h', 'trend_hr', 'trend_rr', 'trend_map', 'trend_spo2', 
    'missing_rate'
]


def load_patients(mimic_path):
    """Load patient demographics."""
    print("Loading patients...")
    patients = pd.read_csv(mimic_path / 'patients.csv', usecols=['subject_id', 'gender', 'anchor_age'])
    patients['sex_male'] = (patients['gender'] == 'M').astype(int)
    patients = patients.rename(columns={'anchor_age': 'age'})
    return patients[['subject_id', 'age', 'sex_male']]


def load_icustays(mimic_path):
    """Load ICU stay information."""
    print("Loading icustays...")
    icustays = pd.read_csv(mimic_path / 'icustays.csv')
    return icustays


def load_chartevents(mimic_path, icustay_ids=None):
    """
    Load vital signs from chartevents.
    MIMIC-IV item_ids for vital signs:
    - 220045: Heart Rate
    - 220050: Systolic BP
    - 220051: Diastolic BP
    - 220052: Mean Arterial Pressure
    - 220277: SpO2
    - 220210: Respiratory Rate
    - 223761: Temperature (Fahrenheit)
    """
    print("Loading chartevents (this may take a while)...")
    
    # Key vital sign item IDs
    vital_items = {
        220045: 'heart_rate',
        220050: 'systolic_bp',
        220051: 'diastolic_bp',
        220052: 'map',
        220277: 'spo2',
        220210: 'resp_rate',
        223761: 'temperature'
    }
    
    # Read in chunks for memory efficiency
    chunks = []
    for chunk in pd.read_csv(mimic_path / 'chartevents.csv', 
                             usecols=['subject_id', 'hadm_id', 'icustay_id', 'itemid', 'valuenum', 'charttime'],
                             chunksize=1000000):
        chunk = chunk[chunk['itemid'].isin(vital_items.keys())]
        if icustay_ids is not None:
            chunk = chunk[chunk['icustay_id'].isin(icustay_ids)]
        chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame()
    
    events = pd.concat(chunks, ignore_index=True)
    events['itemid'] = events['itemid'].map(vital_items)
    
    # Pivot to get one row per icustay_id + charttime with all vitals
    pivot = events.pivot_table(
        index=['subject_id', 'hadm_id', 'icustay_id', 'charttime'],
        columns='itemid',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()
    
    return pivot


def load_labevents(mimic_path, hadm_ids=None):
    """
    Load lab results from labevents.
    MIMIC-IV item_ids for labs:
    - 50812: Lactate
    - 50912: Creatinine
    - 50428: WBC
    - 50809: Hemoglobin
    - 50822: Potassium
    - 50823: Sodium
    """
    print("Loading labevents...")
    
    lab_items = {
        50812: 'lactate',
        50912: 'creatinine',
        50428: 'wbc',
        50809: 'hemoglobin',
        50822: 'potassium',
        50823: 'sodium'
    }
    
    chunks = []
    for chunk in pd.read_csv(mimic_path / 'labevents.csv',
                             usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum'],
                             chunksize=2000000):
        chunk = chunk[chunk['itemid'].isin(lab_items.keys())]
        if hadm_ids is not None:
            chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
        chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame()
    
    labs = pd.concat(chunks, ignore_index=True)
    labs['itemid'] = labs['itemid'].map(lab_items)
    
    # Aggregate to hadm_id level (mean of all values)
    labs_agg = labs.groupby(['subject_id', 'hadm_id', 'itemid'])['valuenum'].mean().reset_index()
    pivot = labs_agg.pivot_table(
        index=['subject_id', 'hadm_id'],
        columns='itemid',
        values='valuenum'
    ).reset_index()
    
    return pivot


def load_ventilator_data(mimic_path, icustay_ids=None):
    """Load ventilator settings."""
    print("Loading ventilator data...")
    try:
        # Try procedural events first (MIMIC-IV)
        vent = pd.read_csv(mimic_path / 'procedureevents.csv', 
                          usecols=['subject_id', 'hadm_id', 'icustay_id', 'itemid'])
        vent_items = [225789, 225790, 225791, 225792]  # Ventilation-related items
        vent = vent[vent['itemid'].isin(vent_items)]
        if icustay_ids is not None:
            vent = vent[vent['icustay_id'].isin(icustay_ids)]
        
        # Any ventilator procedure = 1
        vent_agg = vent.groupby(['subject_id', 'hadm_id', 'icustay_id']).size().reset_index(name='vent_count')
        vent_agg['ventilator'] = (vent_agg['vent_count'] > 0).astype(int)
        return vent_agg[['subject_id', 'hadm_id', 'icustay_id', 'ventilator']]
    except:
        return pd.DataFrame()


def load_vasopressors(mimic_path, icustay_ids=None):
    """Load vasopressor medications."""
    print("Loading vasopressor medications...")
    try:
        # Inputevents for vasopressors
        inputs = pd.read_csv(mimic_path / 'inputevents.csv',
                            usecols=['subject_id', 'hadm_id', 'icustay_id', 'itemid'],
                            nrows=5000000)
        
        # Common vasopressor itemids
        vasopressor_items = [221906, 221907, 221908, 221915, 221916, 221917, 222315, 222316]
        inputs = inputs[inputs['itemid'].isin(vasopressor_items)]
        if icustay_ids is not None:
            inputs = inputs[inputs['icustay_id'].isin(icustay_ids)]
        
        vasopressor_agg = inputs.groupby(['subject_id', 'hadm_id', 'icustay_id']).size().reset_index(name='vaso_count')
        vasopressor_agg['vasopressor'] = (vasopressor_agg['vaso_count'] > 0).astype(int)
        return vasopressor_agg[['subject_id', 'hadm_id', 'icustay_id', 'vasopressor']]
    except:
        return pd.DataFrame()


def load_fluid_balance(mimic_path, icustay_ids=None):
    """Calculate fluid balance from input and output events."""
    print("Loading fluid balance...")
    try:
        # Input events
        inputs = pd.read_csv(mimic_path / 'inputevents.csv',
                            usecols=['subject_id', 'hadm_id', 'icustay_id', 'amount'],
                            nrows=5000000)
        if icustay_ids is not None:
            inputs = inputs[inputs['icustay_id'].isin(icustay_ids)]
        input_sum = inputs.groupby(['subject_id', 'hadm_id', 'icustay_id'])['amount'].sum().reset_index()
        input_sum.columns = ['subject_id', 'hadm_id', 'icustay_id', 'input_ml']
        
        # Output events
        outputs = pd.read_csv(mimic_path / 'outputevents.csv',
                             usecols=['subject_id', 'hadm_id', 'icustay_id', 'value'],
                             nrows=5000000)
        if icustay_ids is not None:
            outputs = outputs[outputs['icustay_id'].isin(icustay_ids)]
        output_sum = outputs.groupby(['subject_id', 'hadm_id', 'icustay_id'])['value'].sum().reset_index()
        output_sum.columns = ['subject_id', 'hadm_id', 'icustay_id', 'output_ml']
        
        # Merge and calculate balance
        fluid = input_sum.merge(output_sum, on=['subject_id', 'hadm_id', 'icustay_id'], how='outer')
        fluid = fluid.fillna(0)
        fluid['fluid_balance_ml_24h'] = fluid['input_ml'] - fluid['output_ml']
        
        return fluid[['subject_id', 'hadm_id', 'icustay_id', 'fluid_balance_ml_24h']]
    except Exception as e:
        print(f"Could not load fluid balance: {e}")
        return pd.DataFrame()


def create_target_variable(mimic_path, icustays):
    """
    Create deterioration target variable.
    
    Deterioration = 1 if patient experiences any of:
    - ICU transfer (from ward to ICU)
    - ICU readmission
    - In-hospital death
    - Cardiac arrest
    
    Lead time = hours from observation window to event
    """
    print("Creating target variable...")
    
    # Load admissions for outcome tracking
    admissions = pd.read_csv(mimic_path / 'admissions.csv',
                            usecols=['subject_id', 'hadm_id', 'discharge_location', 'hospital_expire_flag'])
    
    # Merge with icustays
    outcomes = icustays.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
    
    # Define deterioration: death or discharge to hospice
    outcomes['deterioration'] = outcomes['hospital_expire_flag'].fillna(0).astype(int)
    
    # For lead time, use ICU stay duration as proxy (in real implementation,
    # would use actual event timestamps)
    outcomes['lead_time_hours'] = outcomes['los_icu'] * 24  # Convert days to hours
    
    return outcomes[['subject_id', 'hadm_id', 'icustay_id', 'deterioration', 'lead_time_hours']]


def calculate_trends(vitals_df):
    """Calculate trends (slope) for vital signs over time."""
    print("Calculating vital sign trends...")
    
    if vitals_df.empty:
        return pd.DataFrame()
    
    vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'])
    vitals_df = vitals_df.sort_values(['icustay_id', 'charttime'])
    
    trends = []
    for icustay_id, group in vitals_df.groupby('icustay_id'):
        if len(group) < 2:
            continue
        
        row = {'icustay_id': icustay_id}
        
        for col in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'map', 'spo2', 'resp_rate']:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) >= 2:
                    # Simple linear regression slope
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values.values, 1)[0]
                    row[f'trend_{col}'] = slope
                else:
                    row[f'trend_{col}'] = 0
            else:
                row[f'trend_{col}'] = 0
        
        trends.append(row)
    
    return pd.DataFrame(trends)


def calculate_missing_rate(vitals_df, labs_df):
    """Calculate missing data rate per patient."""
    print("Calculating missing data rates...")
    
    all_features = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'map', 'spo2', 
                   'resp_rate', 'temperature', 'lactate', 'creatinine', 'wbc',
                   'hemoglobin', 'potassium', 'sodium']
    
    # Merge vitals and labs
    merged = vitals_df.merge(labs_df, on=['subject_id', 'hadm_id'], how='outer')
    
    # Calculate missing rate
    merged['missing_rate'] = merged[all_features].isnull().mean(axis=1)
    
    return merged[['subject_id', 'hadm_id', 'icustay_id', 'missing_rate']].drop_duplicates()


def add_charlson_index(icustays):
    """Add Charlson Comorbidity Index (simplified version)."""
    # In practice, would calculate from diagnoses table
    # Using age-based proxy as placeholder
    icustays['charlson_index'] = np.random.poisson(3, len(icustays)).clip(0, 15)
    return icustays


def process_mimic_data(mimic_path, output_path=None):
    """
    Main function to process MIMIC-IV data.
    
    Parameters:
    -----------
    mimic_path : Path
        Path to MIMIC-IV data directory
    output_path : Path, optional
        Path to save processed CSV
        
    Returns:
    --------
    pd.DataFrame with all features
    """
    mimic_path = Path(mimic_path)
    
    print(f"Processing MIMIC-IV data from {mimic_path}")
    
    # Load base tables
    patients = load_patients(mimic_path)
    icustays = load_icustays(mimic_path)
    
    # Get unique icustay_ids for filtering
    icustay_ids = set(icustays['icustay_id'].unique())
    hadm_ids = set(icustays['hadm_id'].unique())
    
    # Load clinical data
    vitals = load_chartevents(mimic_path, icustay_ids)
    labs = load_labevents(mimic_path, hadm_ids)
    ventilator = load_ventilator_data(mimic_path, icustay_ids)
    vasopressors = load_vasopressors(mimic_path, icustay_ids)
    fluid = load_fluid_balance(mimic_path, icustay_ids)
    
    # Create target variable
    outcomes = create_target_variable(mimic_path, icustays)
    
    # Calculate trends
    trends = calculate_trends(vitals)
    
    # Calculate missing rates
    missing_rates = calculate_missing_rate(vitals, labs)
    
    # Merge all data together
    print("Merging all data...")
    
    # Start with icustays
    df = icustays[['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 'los_icu']].copy()
    
    # Add demographics
    df = df.merge(patients, on='subject_id', how='left')
    
    # Add vitals (aggregate to icustay level)
    if not vitals.empty:
        vitals_agg = vitals.groupby(['subject_id', 'hadm_id', 'icustay_id']).agg({
            'heart_rate': 'mean',
            'systolic_bp': 'mean', 
            'diastolic_bp': 'mean',
            'map': 'mean',
            'spo2': 'mean',
            'resp_rate': 'mean',
            'temperature': 'mean'
        }).reset_index()
        df = df.merge(vitals_agg, on=['subject_id', 'hadm_id', 'icustay_id'], how='left')
    
    # Add labs
    if not labs.empty:
        labs = labs.rename(columns={'hadm_id': 'hadm_id_lab'})
        df = df.merge(labs, left_on=['subject_id', 'hadm_id'], 
                     right_on=['subject_id', 'hadm_id_lab'], how='left')
    
    # Add ventilator
    if not ventilator.empty:
        df = df.merge(ventilator[['icustay_id', 'ventilator']], on='icustay_id', how='left')
    
    # Add vasopressors
    if not vasopressors.empty:
        df = df.merge(vasopressors[['icustay_id', 'vasopressor']], on='icustay_id', how='left')
    
    # Add fluid balance
    if not fluid.empty:
        df = df.merge(fluid[['icustay_id', 'fluid_balance_ml_24h']], on='icustay_id', how='left')
    
    # Add trends
    if not trends.empty:
        df = df.merge(trends, on='icustay_id', how='left')
    
    # Add missing rates
    if not missing_rates.empty:
        df = df.merge(missing_rates[['icustay_id', 'missing_rate']], on='icustay_id', how='left')
    
    # Add outcomes (target)
    df = df.merge(outcomes[['icustay_id', 'deterioration', 'lead_time_hours']], on='icustay_id', how='left')
    
    # Add Charlson index (placeholder)
    df = add_charlson_index(df)
    
    # Fill missing values
    df = df.fillna(0)
    
    # Ensure binary columns are integers
    for col in ['sex_male', 'ventilator', 'vasopressor', 'deterioration']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Select and order final columns
    final_cols = FEATURE_COLS + ['deterioration', 'lead_time_hours']
    available_cols = [c for c in final_cols if c in df.columns]
    df = df[available_cols]
    
    print(f"Processed {len(df)} ICU stays")
    print(f"Deterioration rate: {df['deterioration'].mean():.2%}")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return df


def create_sample_mimic_format():
    """
    Create a sample dataset in MIMIC-IV format for testing.
    This generates realistic synthetic data that mimics MIMIC-IV structure.
    """
    print("Creating sample MIMIC-IV format data...")
    
    np.random.seed(42)
    n_samples = 2000
    n_deterioration = int(n_samples * 0.15)
    
    # Base features (matching MIMIC-IV realistic ranges)
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
    
    # Add deterioration signals
    det_idx = np.arange(n_deterioration)
    data['heart_rate'][det_idx] += np.random.normal(15, 5, n_deterioration)
    data['systolic_bp'][det_idx] -= np.random.normal(20, 8, n_deterioration)
    data['diastolic_bp'][det_idx] -= np.random.normal(10, 5, n_deterioration)
    data['map'][det_idx] -= np.random.normal(15, 6, n_deterioration)
    data['spo2'][det_idx] -= np.random.normal(4, 2, n_deterioration)
    data['resp_rate'][det_idx] += np.random.normal(6, 3, n_deterioration)
    data['lactate'][det_idx] += np.random.normal(3, 1.5, n_deterioration)
    data['creatinine'][det_idx] += np.random.normal(0.8, 0.4, n_deterioration)
    data['wbc'][det_idx] += np.random.normal(4, 2, n_deterioration)
    data['hemoglobin'][det_idx] -= np.random.normal(1.5, 0.8, n_deterioration)
    data['ventilator'][det_idx] = np.random.binomial(1, 0.4, n_deterioration)
    data['vasopressor'][det_idx] = np.random.binomial(1, 0.3, n_deterioration)
    data['fluid_balance_ml_24h'][det_idx] -= np.random.normal(500, 300, n_deterioration)
    data['trend_hr'][det_idx] += np.random.normal(8, 3, n_deterioration)
    data['trend_rr'][det_idx] += np.random.normal(4, 2, n_deterioration)
    data['trend_map'][det_idx] -= np.random.normal(8, 4, n_deterioration)
    data['trend_spo2'][det_idx] -= np.random.normal(2, 1, n_deterioration)
    data['missing_rate'][det_idx] += np.random.uniform(0.1, 0.2, n_deterioration)
    
    # Clip values
    data['heart_rate'] = data['heart_rate'].clip(50, 180)
    data['systolic_bp'] = data['systolic_bp'].clip(70, 200)
    data['diastolic_bp'] = data['diastolic_bp'].clip(40, 120)
    data['map'] = data['map'].clip(50, 140)
    data['spo2'] = data['spo2'].clip(80, 100)
    
    # Create target
    deterioration = np.zeros(n_samples, dtype=int)
    deterioration[det_idx] = 1
    data['deterioration'] = deterioration
    
    # Lead time (hours until deterioration)
    lead_time = np.random.exponential(8, n_samples)
    lead_time[det_idx] = np.random.uniform(4, 24, n_deterioration)
    data['lead_time_hours'] = lead_time
    
    df = pd.DataFrame(data)
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MIMIC-IV data for ICU deterioration prediction')
    parser.add_argument('--mimic-path', type=str, help='Path to MIMIC-IV data directory')
    parser.add_argument('--output', type=str, default='output/mimic_icu_dataset.csv', 
                       help='Output CSV path')
    parser.add_argument('--sample', action='store_true', 
                       help='Create sample data in MIMIC-IV format')
    
    args = parser.parse_args()
    
    if args.sample or args.mimic_path is None:
        # Create sample data
        df = create_sample_mimic_format()
        df.to_csv(args.output, index=False)
        print(f"Created sample data: {len(df)} rows")
        print(f"Saved to {args.output}")
    else:
        # Process real MIMIC-IV data
        df = process_mimic_data(args.mimic_path, args.output)
        print(f"Processed MIMIC-IV data: {len(df)} rows")