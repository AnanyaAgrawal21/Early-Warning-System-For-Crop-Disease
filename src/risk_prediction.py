import joblib
import pandas as pd

from src.config import MODEL

from src.feature_engineering import (
    add_features,
    FEATURES
)


def load_model():

    if not MODEL.exists():

        raise FileNotFoundError(
            "Train first: python -m src.train_model"
        )

    return joblib.load(MODEL)


def predict_risk(df, model=None):

    model = model or load_model()

    d = add_features(df).copy()

    d["risk_probability"] = (
        model.predict_proba(d[FEATURES])[:, 1]
    )

    d["risk_level"] = pd.cut(
        d.risk_probability,
        [-0.001, 0.35, 0.65, 1.0],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    ).astype(str)

    d["recommended_action"] = (
        d.risk_level.map({

            "Low":
            "Continue routine monitoring",

            "Medium":
            "Inspect this zone soon and recheck sensor readings",

            "High":
            "Prioritize field inspection and seek expert verification"

        })
    )

    return d