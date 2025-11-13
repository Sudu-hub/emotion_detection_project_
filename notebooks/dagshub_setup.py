import dagshub
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/sudarshansahane1044/emotion_detection_project_.mlflow")
dagshub.init(repo_owner='sudarshansahane1044', repo_name='emotion_detection_project_', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)