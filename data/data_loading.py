import os
import joblib
import mlflow.data
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"

ohe_path = ARTIFACTS_DIR / "ohe_encoder.pkl"
ordinal_path = ARTIFACTS_DIR / "ordinal_encoder.pkl"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ai_worker_burnout_attrition_2026.csv"


class DataPreparation:
    def __init__(self):
        self.df = None
        self.ohe_encoder = None
        self.ordinal_encoder = None

    def load_data(self):
        self.df = pd.read_csv(DATA_PATH)
        self.df.drop(columns=["employee_id"], inplace=True)
        return self.df

    def encode_features(self):
        ohe_features = [
            "job_role",
            "country",
            "industry",
            "remote_work_type",
            "primary_ai_tool",
            "ai_adoption_stage",
        ]

        ordinal_features = [
            "education_level",
            "company_size",
            "fear_of_ai_replacement",
        ]

        self.ohe_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        ohe_encoded = self.ohe_encoder.fit_transform(self.df[ohe_features])

        ohe_df = pd.DataFrame(
            ohe_encoded,
            columns=self.ohe_encoder.get_feature_names_out(ohe_features),
            index=self.df.index
        )

        self.df = pd.concat(
            [self.df.drop(columns=ohe_features), ohe_df],
            axis=1
        )

        edu_order = ["Bootcamp", "Bachelor", "Master", "Self-taught", "PhD"]
        company_size_order = [
            "Startup (<50)",
            "Small (50-200)",
            "Mid (200-1000)",
            "Large (1000-5000)",
            "Enterprise (5000+)"
        ]
        fear_order = ["Low", "Medium", "High"]

        self.ordinal_encoder = OrdinalEncoder(
            categories=[
                edu_order,
                company_size_order,
                fear_order
            ]
        )

        self.df[ordinal_features] = self.ordinal_encoder.fit_transform(
            self.df[ordinal_features]
        )

        joblib.dump(self.ohe_encoder, ohe_path)
        joblib.dump(self.ordinal_encoder, ordinal_path)

        return self.df

    def split_data(self):
        y = self.df["attrition_risk"]
        X = self.df.drop(columns=["attrition_risk"])

        y = y.map({"Low": 0, "Medium": 1, "High": 2})

        X = X.astype("float64")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        dataset = mlflow.data.from_pandas(
            pd.concat([X, y], axis=1),
            name="ai_worker_burnout_attrition_2026",
            targets="attrition_risk"
        )

        return X_train, X_test, y_train, y_test, dataset

    def main(self):
        self.load_data()
        self.encode_features()
        return self.split_data()


X_train, X_test, y_train, y_test, dataset = DataPreparation().main()

