# Machine-Predictive-Maintenance

This repository contains a complete, modular, and extensible Predictive Maintenance System implemented in Python. It supports dataset ingestion, feature engineering, exploratory data analysis, anomaly detection, supervised failure prediction, and maintenance report generation. The system can work with real industrial datasets or synthetically generated data.

Overview

The system is structured around a main class, PredictiveMaintenanceSystem, which encapsulates the complete workflow required to build and evaluate predictive maintenance models. The design supports multiple dataset formats, including CSV, Azure PdM datasets, NASA bearing datasets, and custom column mappings. If no dataset is provided, the system can generate synthetic multivariate time-series data.

Features
1. Dataset Management

Supports dataset types: auto-detected, CSV, Azure predictive maintenance format, NASA bearing data, and custom structured files.

Automatic detection of datetime columns.

Automated fallback to synthetic dataset generation when files are missing.

Custom dataset mapping to required columns when necessary.

2. Feature Engineering

Extraction of temporal features (hour, day of week, month).

Rolling 24-hour statistical features for telemetry metrics.

Deviation-from-mean features for drift detection.

Automated handling of missing data.

3. Exploratory Data Analysis

Statistical summaries of all numeric attributes.

Failure distribution overview.

Time-series visualization of telemetry metrics.

Per-machine visualization support when machine identifiers are available.

4. Anomaly Detection

Implementation using IsolationForest.

Automated marking of anomaly labels and anomaly scores.

Visualization of normal vs anomalous behavior across telemetry metrics.

5. Failure Prediction Models

The system trains and evaluates multiple supervised ML models:

Random Forest Classifier

XGBoost Classifier

Support Vector Machine

Logistic Regression

The best model is automatically selected based on accuracy and ROC-AUC.
Feature importance is extracted and ranked for tree-based methods.

6. Health Prediction

A prediction interface is provided to estimate:

Expected failure type

Confidence score

Probability distribution across all failure classes

7. Maintenance Report Generation

Generates a structured report summarizing:

Latest machine readings

Detected anomalies

Detected failures

Threshold-based warnings for vibration, voltage, rotation speed, and pressure

Reports are generated per machine when machine IDs are present.

Project Structure
Predictive_Maintenance.py        # Core implementation of the system
Predictive_Maintenance.csv       # Optional dataset uploaded by the user

Technologies Used

Python 3.x

NumPy

Pandas

Matplotlib

Seaborn

scikit-learn

XGBoost

Usage
Initialization
pms = PredictiveMaintenanceSystem()

Loading a Dataset
pms.upload_dataset("path/to/dataset.csv", dataset_type="auto")

Generating Synthetic Data (if no dataset)
pms.generate_synthetic_data(n_machines=50, days=180)

Feature Engineering
pms.feature_engineering()

Exploratory Analysis
pms.exploratory_data_analysis()

Anomaly Detection
pms.anomaly_detection()

Training Models
pms.train_failure_prediction_models()

Generating a Maintenance Report
pms.generate_maintenance_report()

Predicting Equipment Health
result = pms.predict_equipment_health(features)

Synthetic Data

Synthetic datasets are generated with:

Multivariate telemetry: voltage, rotation, pressure, vibration

Time drift and noise

Machine aging

Random anomalies and failure labeling

This makes the system suitable for experimentation without needing a real dataset.

Use Cases

Industrial IoT systems

Manufacturing and production monitoring

Predictive maintenance research

Reliability engineering

Real-time equipment health monitoring systems
pms.run_full_pipeline()
5.3. Output and Interpretation
Upon execution, the system outputs detailed reports to the console and generates visual artifacts. 
The critical output is the classification_report, which must be evaluated against industrial requirements,
prioritizing Recall for the minority MachineFailure class to minimize false negatives (missed failures).
