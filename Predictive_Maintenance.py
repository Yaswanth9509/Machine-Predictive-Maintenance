
"""
Predictive Maintenance System
A comprehensive solution for industrial equipment failure prediction
Based on research from Kaggle, Medium, and GitHub references
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import xgboost as xgb

class PredictiveMaintenanceSystem:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = None
        self.datetime_col = None  # Track the datetime column name
        
    def upload_dataset(self, file_path=None, dataset_type='auto'):
        """
        Upload your own dataset for predictive maintenance analysis
        
        Parameters:
        file_path (str): Path to your dataset file or directory
        dataset_type (str): Type of dataset ('auto', 'azure', 'nasa', 'csv', 'custom')
        
        Returns:
        pandas.DataFrame: Loaded dataset
        """
        if file_path is None:
            print("No dataset provided. Using synthetic data for demonstration.")
            print("To use your own data, provide a path to your dataset file.")
            return self.generate_synthetic_data()
        
        if not os.path.exists(file_path):
            print(f"Error: The path '{file_path}' does not exist.")
            print("Using synthetic data instead.")
            return self.generate_synthetic_data()
            
        return self.load_dataset(file_path, dataset_type)
        
    def load_dataset(self, dataset_path=None, dataset_type='auto'):
        """Load external dataset for predictive maintenance"""
        if dataset_path is None:
            return self.generate_synthetic_data()
        
        print(f"Loading dataset from: {dataset_path}")
        
        try:
            # Auto-detect dataset type if set to auto
            if dataset_type == 'auto':
                dataset_type = self.detect_dataset_type(dataset_path)
                print(f"Auto-detected dataset type: {dataset_type}")
            
            if dataset_type == 'azure':
                return self.load_azure_dataset(dataset_path)
            elif dataset_type == 'nasa':
                return self.load_nasa_dataset(dataset_path)
            elif dataset_type == 'csv':
                return self.load_csv_dataset(dataset_path)
            elif dataset_type == 'custom':
                return self.load_custom_dataset(dataset_path)
            else:
                print(f"Unsupported dataset type: {dataset_type}")
                return self.generate_synthetic_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data generation...")
            return self.generate_synthetic_data()
    
    def detect_dataset_type(self, dataset_path):
        """Auto-detect the type of dataset based on file structure"""
        if os.path.isdir(dataset_path):
            # Check if it's an Azure-like dataset with multiple CSV files
            files = os.listdir(dataset_path)
            azure_files = ['PdM_telemetry.csv', 'PdM_errors.csv', 'PdM_maint.csv', 
                          'PdM_failures.csv', 'PdM_machines.csv']
            
            if all(file in files for file in azure_files):
                return 'azure'
            else:
                return 'custom'
        else:
            # Single file - check extension and content
            if dataset_path.endswith('.csv'):
                # Try to read and analyze the CSV
                try:
                    df_sample = pd.read_csv(dataset_path, nrows=5)
                    # Check for NASA-like columns
                    nasa_cols = ['timestamp', 'bearing_id', 'voltage', 'rotation']
                    if any(col in df_sample.columns for col in nasa_cols):
                        return 'nasa'
                    else:
                        return 'csv'
                except:
                    return 'csv'
            else:
                return 'custom'
    
    def load_csv_dataset(self, dataset_path):
        """Load a standard CSV dataset for predictive maintenance"""
        print("Loading CSV dataset...")
        data = pd.read_csv(dataset_path)
        self.data = data
        
        # Try to identify datetime column
        self._identify_datetime_column()
        
        print(f"Loaded CSV dataset: {len(self.data)} records")
        print(f"Columns: {list(self.data.columns)}")
        return self.data
    
    def _identify_datetime_column(self):
        """Identify the datetime column in the dataset"""
        datetime_candidates = ['datetime', 'date', 'time', 'timestamp']
        
        for col in self.data.columns:
            if col.lower() in datetime_candidates:
                self.datetime_col = col
                print(f"Identified datetime column: {self.datetime_col}")
                return
        
        # If no datetime column found, check for any date-like column
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    pd.to_datetime(self.data[col].head())
                    self.datetime_col = col
                    print(f"Identified datetime column: {self.datetime_col}")
                    return
                except:
                    continue
        
        # If still no datetime column, create one
        print("No datetime column found. Creating a synthetic datetime column...")
        self.datetime_col = 'datetime'
        self.data[self.datetime_col] = pd.date_range(
            start='2023-01-01', 
            periods=len(self.data), 
            freq='H'
        )
    
    def load_azure_dataset(self, dataset_path):
        """Load Azure Predictive Maintenance dataset format"""
        print("Loading Azure Predictive Maintenance dataset...")
        # This would be the implementation for Azure dataset
        # For now, we'll just load the telemetry data
        telemetry_path = os.path.join(dataset_path, 'PdM_telemetry.csv')
        if os.path.exists(telemetry_path):
            self.data = pd.read_csv(telemetry_path)
            
            # Try to identify datetime column
            self._identify_datetime_column()
            
            print(f"Loaded Azure dataset: {len(self.data)} records")
            return self.data
        else:
            print("Azure dataset files not found in the specified directory")
            return self.generate_synthetic_data()
    
    def load_nasa_dataset(self, dataset_path):
        """Load NASA bearing dataset format"""
        print("Loading NASA bearing dataset...")
        # This would be the implementation for NASA dataset
        # For now, we'll just load the CSV directly
        self.data = pd.read_csv(dataset_path)
        
        # Try to identify datetime column
        self._identify_datetime_column()
        
        print(f"Loaded NASA dataset: {len(self.data)} records")
        return self.data
    
    def load_custom_dataset(self, dataset_path):
        """Load a custom dataset with flexible column mapping"""
        print("Loading custom dataset...")
        
        # Load the data
        if dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
            data = pd.read_excel(dataset_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        print("Dataset columns detected:", list(data.columns))
        
        # Ask user to map columns if needed
        required_cols = ['datetime', 'machineID', 'failure']
        telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
        
        # Check if we have the required columns
        missing_required = [col for col in required_cols if col not in data.columns]
        missing_telemetry = [col for col in telemetry_cols if col not in data.columns]
        
        if missing_required or len(missing_telemetry) > 2:
            print("\nYour dataset is missing some recommended columns:")
            if missing_required:
                print("Required columns:", missing_required)
            if missing_telemetry:
                print("Recommended telemetry columns:", missing_telemetry)
            
            print("\nPlease map your columns to the expected format:")
            column_mapping = {}
            
            for col in required_cols + telemetry_cols:
                if col not in data.columns:
                    print(f"\nAvailable columns: {list(data.columns)}")
                    mapped_col = input(f"Which column should be mapped to '{col}'? (press Enter to skip): ")
                    if mapped_col and mapped_col in data.columns:
                        column_mapping[col] = mapped_col
                    else:
                        print(f"Skipping {col}")
            
            # Apply the mapping
            data = data.rename(columns=column_mapping)
        
        # Ensure we have at least machineID and datetime
        if 'machineID' not in data.columns:
            data['machineID'] = 1  # Default machine ID
        
        if 'datetime' not in data.columns:
            # Try to find a date/time column
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                data = data.rename(columns={date_cols[0]: 'datetime'})
            else:
                # Create a datetime column if none exists
                data['datetime'] = pd.date_range(start='2023-01-01', periods=len(data), freq='H')
        
        # Ensure failure column exists
        if 'failure' not in data.columns:
            data['failure'] = 'none'  # Default to no failures
        
        # Fill missing telemetry columns with default values
        for col in telemetry_cols:
            if col not in data.columns:
                data[col] = 0.0
        
        # Convert datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
            data['datetime'] = pd.to_datetime(data['datetime'])
        
        self.data = data
        self.datetime_col = 'datetime'
        print(f"Loaded custom dataset: {len(self.data)} records")
        if 'machineID' in self.data.columns:
            print(f"Machines: {self.data['machineID'].nunique()}")
        
        return self.data

    def generate_synthetic_data(self, n_machines=10, days=30):
        """Generate synthetic predictive maintenance data for demonstration"""
        print("Generating synthetic predictive maintenance data...")
        
        # Create date range
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days*24, freq='H')
        
        # Create synthetic data
        data = []
        for machine_id in range(1, n_machines + 1):
            # Base values for each machine
            base_volt = np.random.uniform(170, 190)
            base_rotate = np.random.uniform(1500, 1700)
            base_pressure = np.random.uniform(100, 110)
            base_vibration = np.random.uniform(40, 50)
            
            # Machine age (in days)
            age = np.random.randint(30, 365*3)
            
            for i, date in enumerate(dates):
                # Add some random noise and drift over time
                drift_factor = 1 + (i / (len(dates) * 2))  # Small drift over time
                
                volt = base_volt * drift_factor + np.random.normal(0, 2)
                rotate = base_rotate * drift_factor + np.random.normal(0, 20)
                pressure = base_pressure * drift_factor + np.random.normal(0, 1)
                vibration = base_vibration * drift_factor + np.random.normal(0, 0.5)
                
                # Add some anomalies
                if np.random.random() < 0.01:  # 1% chance of anomaly
                    anomaly_type = np.random.choice(['volt', 'rotate', 'pressure', 'vibration'])
                    if anomaly_type == 'volt':
                        volt *= np.random.uniform(1.2, 1.5)
                    elif anomaly_type == 'rotate':
                        rotate *= np.random.uniform(0.7, 0.9)
                    elif anomaly_type == 'pressure':
                        pressure *= np.random.uniform(1.3, 1.8)
                    elif anomaly_type == 'vibration':
                        vibration *= np.random.uniform(1.5, 2.5)
                
                # Determine failure based on conditions
                failure = 'none'
                if vibration > 70:
                    failure = 'vibration_failure'
                elif volt > 220:
                    failure = 'power_failure'
                elif rotate < 1200:
                    failure = 'rotation_failure'
                elif pressure > 130:
                    failure = 'pressure_failure'
                
                data.append({
                    'datetime': date,
                    'machineID': machine_id,
                    'volt': volt,
                    'rotate': rotate,
                    'pressure': pressure,
                    'vibration': vibration,
                    'age': age + i/24,  # Age increases with time
                    'failure': failure
                })
        
        self.data = pd.DataFrame(data)
        self.datetime_col = 'datetime'
        print(f"Generated synthetic data: {len(self.data)} records")
        print(f"Machines: {self.data['machineID'].nunique()}")
        return self.data

    def feature_engineering(self):
        """Create additional features for predictive maintenance"""
        if self.data is None:
            print("No data available. Please load or generate data first.")
            return
        
        print("Performing feature engineering...")
        
        # Ensure datetime column is identified
        if self.datetime_col is None:
            self._identify_datetime_column()
        
        # Ensure datetime is in proper format
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.datetime_col]):
            self.data[self.datetime_col] = pd.to_datetime(self.data[self.datetime_col])
        
        # Sort by machine (if available) and datetime
        sort_columns = [self.datetime_col]
        if 'machineID' in self.data.columns:
            sort_columns.insert(0, 'machineID')
        
        self.data = self.data.sort_values(sort_columns)
        
        # Create time-based features
        self.data['hour'] = self.data[self.datetime_col].dt.hour
        self.data['day_of_week'] = self.data[self.datetime_col].dt.dayofweek
        self.data['month'] = self.data[self.datetime_col].dt.month
        
        # Create rolling statistics for each machine (if machineID exists)
        if 'machineID' in self.data.columns:
            for machine_id in self.data['machineID'].unique():
                machine_mask = self.data['machineID'] == machine_id
                
                for col in ['volt', 'rotate', 'pressure', 'vibration']:
                    if col in self.data.columns:
                        # Rolling mean and std
                        self.data.loc[machine_mask, f'{col}_mean_24h'] = (
                            self.data.loc[machine_mask, col].rolling(window=24, min_periods=1).mean()
                        )
                        self.data.loc[machine_mask, f'{col}_std_24h'] = (
                            self.data.loc[machine_mask, col].rolling(window=24, min_periods=1).std()
                        )
                        
                        # Difference from mean
                        self.data.loc[machine_mask, f'{col}_diff_from_mean'] = (
                            self.data.loc[machine_mask, col] - self.data.loc[machine_mask, f'{col}_mean_24h']
                        )
        else:
            # If no machineID, calculate rolling stats for the entire dataset
            for col in ['volt', 'rotate', 'pressure', 'vibration']:
                if col in self.data.columns:
                    # Rolling mean and std
                    self.data[f'{col}_mean_24h'] = (
                        self.data[col].rolling(window=24, min_periods=1).mean()
                    )
                    self.data[f'{col}_std_24h'] = (
                        self.data[col].rolling(window=24, min_periods=1).std()
                    )
                    
                    # Difference from mean
                    self.data[f'{col}_diff_from_mean'] = (
                        self.data[col] - self.data[f'{col}_mean_24h']
                    )
        
        # Fill NaN values
        self.data = self.data.fillna(method='bfill')
        
        print("Feature engineering completed.")
        return self.data

    def exploratory_data_analysis(self):
        """Perform exploratory data analysis on the dataset"""
        if self.data is None:
            print("No data available. Please load or generate data first.")
            return
        
        print("Performing exploratory data analysis...")
        
        # Basic info
        print(f"\nDataset shape: {self.data.shape}")
        
        if self.datetime_col:
            print(f"Time range: {self.data[self.datetime_col].min()} to {self.data[self.datetime_col].max()}")
        
        if 'machineID' in self.data.columns:
            print(f"Number of machines: {self.data['machineID'].nunique()}")
        
        # Summary statistics
        print("\nSummary statistics:")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_cols].describe())
        
        # Failure distribution
        if 'failure' in self.data.columns:
            print("\nFailure distribution:")
            print(self.data['failure'].value_counts())
            
            # Plot failure distribution
            plt.figure(figsize=(10, 6))
            self.data['failure'].value_counts().plot(kind='bar')
            plt.title('Failure Type Distribution')
            plt.xlabel('Failure Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # Time series plots for a sample machine or the entire dataset
        if self.datetime_col:
            if 'machineID' in self.data.columns:
                sample_machine = self.data['machineID'].iloc[0]
                machine_data = self.data[self.data['machineID'] == sample_machine]
                plot_title = f'Telemetry Data for Machine {sample_machine}'
                plot_data = machine_data
            else:
                plot_title = 'Telemetry Data'
                plot_data = self.data
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(plot_title)
            
            metrics = ['volt', 'rotate', 'pressure', 'vibration']
            for i, metric in enumerate(metrics):
                if metric in self.data.columns:
                    ax = axes[i//2, i%2]
                    ax.plot(plot_data[self.datetime_col], plot_data[metric])
                    ax.set_title(f'{metric.capitalize()} over Time')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(metric.capitalize())
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        print("Exploratory data analysis completed.")

    def anomaly_detection(self):
        """Detect anomalies in the equipment data"""
        if self.data is None:
            print("No data available. Please load or generate data first.")
            return
        
        print("Performing anomaly detection...")
        
        # Select features for anomaly detection
        features = [col for col in ['volt', 'rotate', 'pressure', 'vibration'] if col in self.data.columns]
        
        if not features:
            print("No telemetry features found for anomaly detection.")
            return
        
        X = self.data[features].values
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        
        # Add anomaly labels to data
        self.data['anomaly'] = anomalies
        self.data['anomaly_score'] = iso_forest.decision_function(X_scaled)
        
        # Count anomalies
        n_anomalies = (anomalies == -1).sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(anomalies)*100:.2f}% of data)")
        
        # Plot anomalies if datetime column exists
        if self.datetime_col:
            if 'machineID' in self.data.columns:
                sample_machine = self.data['machineID'].iloc[0]
                plot_data = self.data[self.data['machineID'] == sample_machine]
                plot_title = f'Anomaly Detection for Machine {sample_machine}'
            else:
                plot_data = self.data
                plot_title = 'Anomaly Detection'
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(plot_title)
            
            metrics = [col for col in ['volt', 'rotate', 'pressure', 'vibration'] if col in self.data.columns]
            for i, metric in enumerate(metrics):
                if i < 4:  # Only plot up to 4 metrics
                    ax = axes[i//2, i%2]
                    normal_data = plot_data[plot_data['anomaly'] == 1]
                    anomaly_data = plot_data[plot_data['anomaly'] == -1]
                    
                    ax.plot(normal_data[self.datetime_col], normal_data[metric], 'b-', label='Normal', alpha=0.7)
                    ax.scatter(anomaly_data[self.datetime_col], anomaly_data[metric], color='red', label='Anomaly')
                    ax.set_title(f'{metric.capitalize()} with Anomalies')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(metric.capitalize())
                    ax.legend()
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        print("Anomaly detection completed.")

    def train_failure_prediction_models(self):
        """Train machine learning models for failure prediction"""
        if self.data is None:
            print("No data available. Please load or generate data first.")
            return
        
        print("Training failure prediction models...")
        
        # Prepare data for training
        if 'failure' not in self.data.columns:
            print("No failure column found in data. Cannot train prediction models.")
            return
        
        # Encode failure types
        label_encoder = LabelEncoder()
        self.data['failure_encoded'] = label_encoder.fit_transform(self.data['failure'])
        
        # Select features for training
        feature_cols = [col for col in self.data.columns if col not in [
            self.datetime_col, 'machineID', 'failure', 'failure_encoded', 'anomaly', 'anomaly_score'
        ] and self.data[col].dtype in [np.int64, np.float64]]
        
        if not feature_cols:
            print("No suitable features found for training.")
            return
        
        X = self.data[feature_cols].values
        y = self.data['failure_encoded'].values
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['failure_prediction'] = scaler
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            
            accuracy = model.score(X_test_scaled, y_test)
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"{name} ROC AUC: {roc_auc:.4f}")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
            
            # Store the best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
        
        # Store the best model
        self.models['failure_prediction'] = best_model
        self.models['failure_encoder'] = label_encoder
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 most important features:")
            print(self.feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=self.feature_importance.head(15))
            plt.title('Top 15 Feature Importance')
            plt.tight_layout()
            plt.show()
        
        print("Model training completed.")

    def predict_equipment_health(self, features):
        """Predict equipment health based on current features"""
        if 'failure_prediction' not in self.models:
            print("No trained model available. Please train models first.")
            return None
        
        # Scale features
        scaler = self.scalers['failure_prediction']
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        model = self.models['failure_prediction']
        label_encoder = self.models['failure_encoder']
        
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        predicted_failure = label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities[prediction[0]]
        
        # Create result dictionary
        result = {
            'predicted_failure': predicted_failure,
            'confidence': confidence,
            'probabilities': dict(zip(label_encoder.classes_, probabilities))
        }
        
        return result

    def generate_maintenance_report(self):
        """Generate a maintenance report based on predictions"""
        if self.data is None:
            print("No data available. Please load or generate data first.")
            return
        
        print("Generating maintenance report...")
        
        # Get the latest data for each machine (if machineID exists)
        if 'machineID' in self.data.columns:
            if self.datetime_col:
                latest_data = self.data.sort_values(self.datetime_col).groupby('machineID').last().reset_index()
            else:
                latest_data = self.data.groupby('machineID').last().reset_index()
            
            print("\nMAINTENANCE REPORT")
            print("=" * 50)
            
            for _, machine in latest_data.iterrows():
                print(f"\nMachine {machine['machineID']}:")
                
                # Check for anomalies
                if 'anomaly' in machine and machine['anomaly'] == -1:
                    print(" ANOMALY DETECTED - Requires immediate inspection")
                    if 'anomaly_score' in machine:
                        print(f"  Anomaly score: {machine['anomaly_score']:.4f}")
                
                # Check for failures
                if 'failure' in machine and machine['failure'] != 'none':
                    print(f" FAILURE DETECTED: {machine['failure']}")
                
                # Check sensor values
                if 'vibration' in machine and machine['vibration'] > 60:
                    print("  High vibration detected - Check bearings and alignment")
                if 'volt' in machine and machine['volt'] > 210:
                    print("  High voltage detected - Check power supply")
                if 'rotate' in machine and machine['rotate'] < 1300:
                    print("  Low rotation speed detected - Check motor and load")
                if 'pressure' in machine and machine['pressure'] > 120:
                    print("  High pressure detected - Check hydraulic system")
                
                # If no issues found
                if not any([key in machine for key in ['anomaly', 'failure']]) or (
                    'anomaly' in machine and machine['anomaly'] == 1 and 
                    'failure' in machine and machine['failure'] == 'none'
                ):
                    print(" No issues detected - Operating normally")
        else:
            # If no machineID, just report on the overall dataset
            print("\nMAINTENANCE REPORT")
            print("=" * 50)
            
            # Check for anomalies
            if 'anomaly' in self.data.columns:
                n_anomalies = (self.data['anomaly'] == -1).sum()
                if n_anomalies > 0:
                    print(f"{n_anomalies} ANOMALIES DETECTED - Requires inspection")
            
            # Check for failures
            if 'failure' in self.data.columns:
                failures = self.data[self.data['failure'] != 'none']
                if len(failures) > 0:
                    print(f" {len(failures)} FAILURES DETECTED")
                    print("  Failure types:")
                    for failure_type, count in failures['failure'].value_counts().items():
                        print(f"    {failure_type}: {count}")
            
            # If no issues found
            if (('anomaly' not in self.data.columns or n_anomalies == 0) and
                ('failure' not in self.data.columns or len(failures) == 0)):
                print(" No issues detected - Operating normally")
        
        print("\nMaintenance report generated.")

def main():
    """Main function to run the predictive maintenance system"""
    print("Initializing Predictive Maintenance System...")
    print("Based on research from Kaggle, Medium, and GitHub references")
    print("-" * 60)
    
    # Initialize system
    pms = PredictiveMaintenanceSystem()
    
    # DIRECT PATH TO YOUR DATASET - MODIFY THIS LINE
    dataset_path = "C:/Users/yaswa/OneDrive/Desktop/ai4i2020.csv"  # Change this to your actual file path
    
    if dataset_path and os.path.exists(dataset_path):
        dataset_type = "auto"  # Let the system auto-detect, or specify "csv", "nasa", etc.
        pms.upload_dataset(dataset_path, dataset_type)
    else:
        # Use synthetic data
        print("Dataset path not found. Generating synthetic data for demonstration...")
        pms.generate_synthetic_data(n_machines=50, days=180)
    
    # Feature engineering
    pms.feature_engineering()
    
    # Exploratory Data Analysis
    pms.exploratory_data_analysis()
    
    # Anomaly Detection
    pms.anomaly_detection()
    
    # Train failure prediction models
    pms.train_failure_prediction_models()
    
    # Generate maintenance report
    pms.generate_maintenance_report()
    
    # Demo: Predict for a sample machine
    print("\nSAMPLE PREDICTION:")
    print("-" * 30)
    
    # Create sample input (representing current machine state)
    sample_features = np.array([
        180, 1600, 105, 45, 8.5,  # volt, rotate, pressure, vibration, age
        1, 0, 1, 0, 0,  # errors 1-5
        15, 25, 30, 45,  # days since component replacement
        2,  # model_encoded
        # Include engineered features (simplified)
        *[np.random.normal() for _ in range(28)]  # placeholder for lag features
    ])
    
    prediction_result = pms.predict_equipment_health(sample_features)
    if prediction_result:
        print(f"Predicted Failure Type: {prediction_result['predicted_failure']}")
        print(f"Confidence: {prediction_result['confidence']:.2%}")
        print("Component Failure Probabilities:")
        for component, prob in prediction_result['probabilities'].items():
            print(f"  {component}: {prob:.2%}")
    
    print("\nPredictive Maintenance Analysis Complete!")
    print("Models trained and ready for real-time monitoring")

if __name__ == "__main__":
    main()