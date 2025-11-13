# model_evaluation.py (updated)
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from datetime import datetime

# Set up MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/sudarshansahane1044/emotion_detection_project_.mlflow')
dagshub.init(repo_owner='sudarshansahane1044', repo_name='emotion_detection_project_', mlflow=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers if script is imported multiple times
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
else:
    # ensure handlers exist and set levels/formatters
    logger.handlers = [console_handler, file_handler]

def ensure_reports_dir():
    os.makedirs('reports', exist_ok=True)
    logger.debug('Ensured reports/ directory exists')

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)

        # If classifier supports predict_proba, use the positive class probability for AUC.
        if hasattr(clf, 'predict_proba'):
            try:
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
            except Exception:
                # if predict_proba exists but fails (e.g., single-class), fallback to predictions
                y_pred_proba = y_pred
                logger.warning('predict_proba failed; falling back to deterministic predictions for AUC')
        elif hasattr(clf, 'decision_function'):
            try:
                y_pred_proba = clf.decision_function(X_test)
            except Exception:
                y_pred_proba = y_pred
                logger.warning('decision_function failed; falling back to deterministic predictions for AUC')
        else:
            y_pred_proba = y_pred
            logger.warning('Classifier has no predict_proba/decision_function; using discrete predictions for AUC (may be degenerate)')

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            logger.warning('AUC calculation failed: %s. Setting auc to 0.0', e)
            auc = 0.0

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc)
        }
        logger.debug('Model evaluation metrics calculated: %s', metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path,
            'saved_at': datetime.utcnow().isoformat() + 'Z'
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    ensure_reports_dir()
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            metrics_path = 'reports/metrics.json'
            info_path = 'reports/experiment_info.json'

            save_metrics(metrics, metrics_path)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow (if any)
            if hasattr(clf, 'get_params'):
                try:
                    params = clf.get_params()
                    for param_name, param_value in params.items():
                        # mlflow requires param values to be str/int/float
                        mlflow.log_param(param_name, str(param_value))
                except Exception as e:
                    logger.warning('Failed to log model params: %s', e)
            
            # Log model to MLflow
            try:
                mlflow.sklearn.log_model(clf, "model")
            except Exception as e:
                logger.warning('Failed to log model to MLflow: %s', e)

            # Save model info (this is the file DVC expects)
            save_model_info(run.info.run_id, "model", info_path)
            
            # Log the metrics file and experiment info file to MLflow
            try:
                mlflow.log_artifact(metrics_path)
            except Exception as e:
                logger.warning('Failed to log metrics artifact: %s', e)
            try:
                mlflow.log_artifact(info_path)
            except Exception as e:
                logger.warning('Failed to log experiment info artifact: %s', e)

            # Log the evaluation errors log file to MLflow (optional)
            if os.path.exists('model_evaluation_errors.log'):
                try:
                    mlflow.log_artifact('model_evaluation_errors.log')
                except Exception as e:
                    logger.warning('Failed to log evaluation errors log: %s', e)

            logger.info('Model evaluation completed successfully for run_id=%s', run.info.run_id)
        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
