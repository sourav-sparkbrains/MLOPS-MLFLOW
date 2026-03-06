import os
import mlflow
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

REFERENCE_DATA_PATH = PROJECT_DIR / "data" / "ai_worker_burnout_attrition_2026.csv"
NEW_DATA_PATH = PROJECT_DIR / "data" / "new_data.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "drift_report.html"


def check_drift_report():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Production_Monitoring")

    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    ref_df.drop(columns=['employee_id', 'attrition_risk'], axis=1, inplace=True)

    input_df = pd.read_csv(NEW_DATA_PATH)
    input_df.drop(columns=['attrition_risk'], axis=1, inplace=True)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=input_df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))

    result = snapshot.dict()
    first_metric = result["metrics"][0]
    drift_share = first_metric["value"]["share"]
    drift_detected = drift_share > 0.5

    with mlflow.start_run(run_name="data_drift_monitoring"):
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("drift_share", float(drift_share))

        for metric in result["metrics"][1:]:
            column_name = metric["config"]["column"]
            drift_score = float(metric["value"])
            mlflow.log_metric(f"drift_{column_name}", drift_score)

        mlflow.log_artifact(str(REPORT_PATH))

    return drift_detected


if __name__ == "__main__":
    drift = check_drift_report()
    print(f"Drift detected: {drift}")