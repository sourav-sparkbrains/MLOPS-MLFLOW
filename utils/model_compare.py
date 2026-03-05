import mlflow
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

def compare_models():
    try:
        production_model = client.get_model_version_by_alias("BestAttritionModel", "production")
        production_model_run_id = production_model.run_id
        details = client.get_run(production_model_run_id)
        production_accuracy = details.data.metrics["test_accuracy"]
    except Exception:
        production_accuracy = 0.0
        print("No production model found, will promote best run automatically")


    all_runs = mlflow.search_runs(
        experiment_names=["AI_Worker_Burnout_Attrition"],
        order_by=["metrics.test_accuracy DESC"]
    )
    best_new_run = all_runs.iloc[0]
    best_run_id = best_new_run["run_id"]
    best_run_accuracy =  best_new_run["metrics.test_accuracy"]

    print(f"Production accuracy: {production_accuracy:.4f}")
    print(f"Best new run accuracy: {best_run_accuracy:.4f}")

    if best_run_accuracy > production_accuracy + 0.005:
        model_uri = f"runs:/{best_run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name="Best_Ai_Worker_Burnout_Attrition_Classifier")
        client.set_registered_model_alias(
            name="Best_Ai_Worker_Burnout_Attrition_Classifier",
            alias="production",
            version=registered.version
        )
        print(f"Promoted version {registered.version} to production")
    else:
        print("Did not found better model")

if __name__ == "__main__":
    compare_models()
