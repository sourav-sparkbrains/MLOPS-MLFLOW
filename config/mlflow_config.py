import matplotlib
matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
import mlflow.xgboost

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("AI_Worker_Burnout_Attrition")
mlflow.sklearn.autolog()
mlflow.xgboost.autolog()