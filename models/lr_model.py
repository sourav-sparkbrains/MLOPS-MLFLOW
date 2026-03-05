import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_lr_experiment(C, solver,
                      X_train, X_test,
                      y_train, y_test,
                      dataset):
    numeric_cols = [
    'years_experience',
     'team_size',
     'salary_usd_k',
     'ai_tools_used_per_day',
     'ai_replaces_my_tasks_pct',
     'productivity_score',
     'burnout_score']

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols)
        ],
        remainder="passthrough"
    )

    with mlflow.start_run(run_name=f"LR_C{C}_{solver}"):

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=C,
                solver=solver,
                penalty="l2",
                max_iter=3000
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_input(dataset, context="training")

        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        mlflow.sklearn.log_model(pipeline, "logistic_regression_model")

        print(
            f"C={C} | solver={solver} | "
            f"Accuracy: {acc:.4f} | F1: {f1:.4f}"
        )

        return acc