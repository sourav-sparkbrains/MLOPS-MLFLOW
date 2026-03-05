from config import mlflow_config
from data.data_loading import X_train, X_test, y_train, y_test, dataset
from models.dt_model import run_dt_experiment
from models.rf_model import run_rf_experiment
from models.xgboost_model import run_xgb_experiment
from models.lr_model import run_lr_experiment
from models.svm_model import run_svm_experiment


rf_experiments = [
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
]

dt_experiments = [
    {"max_depth": 5},
    {"max_depth": 10},
    {"max_depth": 15},
    {"max_depth": 20},
]

xgb_experiments = [
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    {"n_estimators": 100, "max_depth": 10, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.05},
    {"n_estimators": 200, "max_depth": 15, "learning_rate": 0.05},
]

lr_experiments = [
    {"C": 0.01, "solver": "lbfgs"},
    {"C": 0.1, "solver": "lbfgs"},
    {"C": 1, "solver": "lbfgs"},
    {"C": 1, "solver": "saga"},
]

svm_experiments = [
    {"C": 0.1, "kernel": "linear"},
    {"C": 1, "kernel": "linear"},
    {"C": 1, "kernel": "rbf"},
    {"C": 10, "kernel": "rbf"},
]

def run_all_experiments():
    print("\nRunning Random Forest Experiments")
    for exp in rf_experiments:
        acc = run_rf_experiment(**exp, X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                dataset=dataset)
        print(exp, "->", round(acc, 4))

    print("\nRunning Decision Tree Experiments")
    for exp in dt_experiments:
        acc = run_dt_experiment(**exp, X_train=X_train, X_test=X_test,
                                y_train=y_train, y_test=y_test,
                                dataset=dataset)
        print(exp, "->", round(acc, 4))

    print("\nRunning XGBoost Experiments")
    for exp in xgb_experiments:
        acc = run_xgb_experiment(**exp, X_train=X_train, X_test=X_test,
                                 y_train=y_train, y_test=y_test,
                                 dataset=dataset)
        print(exp, "->", round(acc, 4))

    print("\nRunning Logistic Regression Experiments")
    for exp in lr_experiments:
        acc = run_lr_experiment(**exp,
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test,
                                dataset=dataset)
        print(exp, "->", round(acc, 4))

    print("\nRunning SVM Experiments")
    for exp in svm_experiments:
        acc = run_svm_experiment(**exp,
                                 X_train=X_train,
                                 X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 dataset=dataset)
        print(exp, "->", round(acc, 4))


if __name__ == "__main__":
    run_all_experiments()

