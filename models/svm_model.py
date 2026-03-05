import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer


def run_svm_experiment(C, kernel,
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
        'burnout_score'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols)
        ],
        remainder="passthrough"
    )

    with mlflow.start_run(run_name=f"SVM_C{C}_{kernel}"):

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVC(
                C=C,
                kernel=kernel,
                gamma="scale"
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
        mlflow.log_param("kernel", kernel)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        mlflow.sklearn.log_model(pipeline, "svm_model")

        print(
            f"C={C} | kernel={kernel} | "
            f"Accuracy: {acc:.4f} | F1: {f1:.4f}"
        )

        return acc