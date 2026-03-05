from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow


def run_rf_experiment(n_estimators, max_depth, X_train, y_train, X_test, y_test, dataset):

    with mlflow.start_run(run_name=f"RandomForest_n={n_estimators}_d={max_depth}"):

        mlflow.log_input(dataset, context="training")

        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1
        })

        return acc