from prefect import flow, task
from utils.model_excutor import run_all_experiments
from utils.model_compare import compare_models

@task(log_prints=True)
def train():
    run_all_experiments()

@task(log_prints=True)
def compare_and_promote():
    compare_models()

@flow(name="ml_pipeline")
def ml_pipeline():
    train()
    compare_and_promote()

if __name__ == "__main__":
    ml_pipeline.serve(
        name="ml-pipeline-deployment",
        cron = "0 9 * * 1"
    )

