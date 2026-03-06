import mlflow
from prefect import flow, task
from utils.model_excutor import run_all_experiments
from utils.model_compare import compare_models
from utils.drift_monitor import check_drift_report

mlflow.set_tracking_uri("http://localhost:5000")

@task(log_prints=True)
def train():
    run_all_experiments()

@task(log_prints=True)
def compare_and_promote():
    compare_models()

@task(log_prints=True)
def check_drift():
    try:
        result = check_drift_report()
        print(f"Drift detected: {result}")
        return result
    except Exception as e:
        print(f"Drift check failed: {e}")
        return False

@flow(name="ml_pipeline")
def ml_pipeline():
    drift = check_drift()
    if drift:
        train()
        compare_and_promote()
    else:
        print("No drift detected - skipping retraining")

if __name__ == "__main__":
    ml_pipeline.serve(
        name="ml-pipeline-deployment",
        cron = "0 9 * * 1"
    )

