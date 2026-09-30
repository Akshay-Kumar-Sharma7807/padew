# Personalized Anomaly Detection for Early Warning of ICU Deterioration (PADEW)

## Project Overview
**Title**: Personalized Anomaly Detection for Early Warning of ICU Deterioration

**Objective**: The goal of this project is to develop and validate a personalized anomaly detection system (PADEW) that provides early warnings for ICU deterioration using the MIMIC-IV dataset. The system aims to alert clinicians with a lead time of more than 4 hours before a deterioration event occurs.

## Features
- **Unsupervised Anomaly Detection**: Utilizes unsupervised learning techniques to detect anomalies in patient data.
- **Personalized Explanations with SHAP**: Implements SHAP (SHapley Additive exPlanations) to provide personalized explanations for model predictions, enhancing interpretability and trust in the alerts.
- **Early Warning System**: Generates timely alerts to clinicians, allowing for proactive interventions.
- **Lead Time**: Alerts are designed to provide a lead time of more than 4 hours before potential deterioration events.

## Dataset
The project utilizes the MIMIC-IV database, which includes various tables relevant to ICU patients:
- **patients**: Demographic information (age, sex)
- **admissions**: Hospital admissions data
- **icustays**: ICU stay information
- **chartevents**: Vital signs (heart rate, blood pressure, SpO2, respiratory rate, temperature)
- **labevents**: Lab results (lactate, creatinine, WBC, hemoglobin, potassium, sodium)
- **inputevents**: Fluid input data
- **outputevents**: Fluid output data
- **ventilator_settings**: Mechanical ventilation settings
- **medicationevents**: Vasopressor administration data

## Below is an explanation of the various features used in your Streamlit application for the Personalized Anomaly Detection for Early Warning of ICU Deterioration (PADEW) project, along with their functions and baseline values that are generally considered normal for each patient.

# Feature Explanations and Baseline Values

### Age
- **Function**: Represents the age of the patient, which can influence their physiological response and risk factors for deterioration.
- **Baseline Value**: Typically ranges from 18 to 95 years, with specific thresholds for older adults (e.g., above 65).

### Sex (Male)
- **Function**: A binary feature indicating whether the patient is male (1) or female (0). Gender can affect health outcomes and disease prevalence.
- **Baseline Value**: This feature is binary, so it can either be 0 (female) or 1 (male).

### Charlson Comorbidity Index
- **Function**: A score that predicts the ten-year mortality for a patient who may have a range of comorbid conditions. Higher scores indicate more comorbidities.
- **Baseline Value**: Typically ranges from 0 to 15, with lower values indicating fewer comorbidities.

### Heart Rate
- **Function**: Measures the number of heartbeats per minute. It is a vital sign that can indicate the patient's cardiovascular status.
- **Baseline Value**: Normal resting heart rate is typically between 60 and 100 beats per minute.

### Systolic Blood Pressure
- **Function**: The pressure in the arteries during the contraction of the heart. It is a critical measure of blood flow and heart function.
- **Baseline Value**: Normal systolic blood pressure is usually between 90 and 120 mmHg.

### Diastolic Blood Pressure
- **Function**: The pressure in the arteries when the heart is at rest between beats. Like systolic pressure, it is essential for assessing cardiovascular health.
- **Baseline Value**: Normal diastolic blood pressure is typically between 60 and 80 mmHg.

### Mean Arterial Pressure (MAP)
- **Function**: Represents the average blood pressure in a person’s arteries during one cardiac cycle. It is crucial for ensuring adequate blood flow to organs.
- **Baseline Value**: Normal MAP is usually between 70 and 100 mmHg.

### SpO2 (Oxygen Saturation)
- **Function**: Indicates the percentage of hemoglobin binding sites in the bloodstream occupied by oxygen. It is a vital sign for respiratory health.
- **Baseline Value**: Normal SpO2 levels are typically between 95% and 100%.

### Respiratory Rate
- **Function**: The number of breaths taken per minute. It is an essential indicator of respiratory function.
- **Baseline Value**: Normal respiratory rate is usually between 12 and 20 breaths per minute.

### Temperature
- **Function**: Body temperature is a vital sign that can indicate infection or other health issues.
- **Baseline Value**: Normal body temperature is around 36.1°C to 37.2°C (97°F to 99°F).

### Lactate
- **Function**: A marker of tissue hypoxia and metabolic function. Elevated levels can indicate sepsis or other critical conditions.
- **Baseline Value**: Normal lactate levels are typically less than 2 mmol/L.

### Creatinine
- **Function**: A waste product from muscle metabolism. It is an important indicator of kidney function.
- **Baseline Value**: Normal creatinine levels are generally between 0.6 and 1.2 mg/dL.

### WBC (White Blood Cell Count)
- **Function**: Indicates the immune response. Elevated levels can suggest infection or inflammation.
- **Baseline Value**: Normal WBC count is typically between 4,500 and 11,000 cells per microliter.

### Hemoglobin
- **Function**: A protein in red blood cells that carries oxygen. Low levels can indicate anemia.
- **Baseline Value**: Normal hemoglobin levels are generally between 12 and 17 g/dL for women and 14 to 18 g/dL for men.

### Potassium
- **Function**: An essential electrolyte that is important for heart and muscle function. Abnormal levels can lead to serious health issues.
- **Baseline Value**: Normal potassium levels are typically between 3.5 and 5.0 mEq/L.

### Sodium
- **Function**: An essential electrolyte that helps regulate fluid balance and blood pressure.
- **Baseline Value**: Normal sodium levels are generally between 135 and 145 mEq/L.

### Ventilator
- **Function**: A binary feature indicating whether the patient is on mechanical ventilation (1) or not (0). It is critical for assessing respiratory support needs.
- **Baseline Value**: This feature is binary, so it can either be 0 (not on a ventilator) or 1 (on a ventilator).

### Vasopressor
- **Function**: A binary feature indicating whether the patient is receiving vasopressor medications (1) or not (0). Vasopressors are used to increase blood pressure in critically ill patients.
- **Baseline Value**: This feature is binary, so it can either be 0 (not receiving vasopressors) or 1 (receiving vasopressors).

### Fluid Balance (ml/24h)
- **Function**: Represents the difference between fluid input and output over 24 hours. It is important for assessing fluid status in critically ill patients.
- **Baseline Value**: A normal fluid balance can vary, but a balance close to zero (±500 ml) is often considered stable in ICU settings.

### Trends (Heart Rate, Respiratory Rate, MAP, SpO2)
- **Function**: These features represent the calculated trends (slope) of vital signs over time, indicating whether the values are increasing or decreasing.
- **Baseline Value**: Trends are relative and depend on individual patient baselines; significant deviations from the baseline may indicate deterioration.

### Missing Rate
- **Function**: The proportion of missing data for the patient’s features. High missing rates can affect the reliability of predictions.
- **Baseline Value**: Ideally, the missing rate should be as low as possible, generally below 10% is considered acceptable.
