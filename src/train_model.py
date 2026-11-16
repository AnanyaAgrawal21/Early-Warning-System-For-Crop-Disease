import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.config import MODELS, MODEL, REPORTS

from src.data_collection import load_data

from src.preprocessing import merge_and_clean

from src.feature_engineering import (
    add_features,
    FEATURES
)


def train():

    h, m = load_data()

    d = add_features(
        merge_and_clean(h, m)
    )

    X = d[FEATURES]
    y = d.disease_label

    if y.nunique() < 2:

        raise ValueError(
            "Training data needs both label classes 0 and 1."
        )

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "rf",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )

    ])

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    prob = model.predict_proba(Xte)[:, 1]

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    joblib.dump(model, MODEL)

    report = classification_report(
        yte,
        pred,
        zero_division=0
    )

    (REPORTS / "classification_report.txt").write_text(
        report
    )

    (REPORTS / "metrics.txt").write_text(
        f"Confusion matrix:\n"
        f"{confusion_matrix(yte, pred)}\n"
        f"ROC-AUC: {roc_auc_score(yte, prob):.4f}"
    )

    print(report)

    print("Saved:", MODEL)

    return model


if __name__ == "__main__":

    train()