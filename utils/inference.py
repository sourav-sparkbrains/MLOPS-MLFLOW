import os
import joblib
import mlflow
import requests
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"
LOG_PATH = BASE_DIR.parent / "predictions_log.csv"

ohe_path = ARTIFACTS_DIR / "ohe_encoder.pkl"
ordinal_path = ARTIFACTS_DIR / "ordinal_encoder.pkl"


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Production_Monitoring")

ENDPOINT = "http://127.0.0.1:8080/invocations"
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

ohe_encoder = joblib.load(ohe_path)
ordinal_encoder = joblib.load(ordinal_path)

def do_inference(raw_input):
    data_dict = raw_input.model_dump()

    ohe_features = ["job_role", "country", "industry", "remote_work_type", "primary_ai_tool", "ai_adoption_stage"]
    ordinal_features = ["education_level", "company_size", "fear_of_ai_replacement"]

    df = pd.DataFrame([data_dict])

    ohe_encoded = ohe_encoder.transform(df[ohe_features])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_encoder.get_feature_names_out(ohe_features))

    df[ordinal_features] = ordinal_encoder.transform(df[ordinal_features])

    df = pd.concat([df.drop(columns=ohe_features), ohe_df], axis=1)
    df = df.astype("float64")

    payload = {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df.values.tolist()
        }
    }

    response = requests.post(
        ENDPOINT,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    raw_predictions = response.json()["predictions"]

    mapped_predictions = [LABEL_MAP[p] for p in raw_predictions]

    predictions_df = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()] * len(mapped_predictions),
        "prediction": mapped_predictions
    })


    file_exists = os.path.exists(LOG_PATH)
    predictions_df.to_csv(LOG_PATH,mode="a",index=False,header=not file_exists)

    full_log = pd.read_csv(LOG_PATH)
    rows = full_log.shape[0]

    if rows >= 10 :
        distribution = full_log["prediction"].value_counts(normalize=True)
        with mlflow.start_run(run_name=f"monitoring_batch_{rows}"):
            mlflow.log_metric("low_pct", distribution.get("Low", 0))
            mlflow.log_metric("medium_pct", distribution.get("Medium", 0))
            mlflow.log_metric("high_pct", distribution.get("High", 0))

    print(f"Predicted Attrition Risk: {mapped_predictions}")

    return {"Predicted Attrition Risk": mapped_predictions[0],"status":"success"}