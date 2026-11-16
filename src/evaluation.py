import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)

from src.config import REPORTS

from src.data_collection import load_data

from src.preprocessing import merge_and_clean

from src.feature_engineering import (
    add_features,
    FEATURES
)

from src.risk_prediction import load_model


def evaluate():

    h, m = load_data()

    d = add_features(
        merge_and_clean(h, m)
    )

    model = load_model()

    pred = model.predict(
        d[FEATURES]
    )

    REPORTS.mkdir(exist_ok=True)

    report = classification_report(
        d.disease_label,
        pred,
        zero_division=0
    )

    (
        REPORTS / "evaluation_report.txt"
    ).write_text(report)

    disp = ConfusionMatrixDisplay.from_predictions(
        d.disease_label,
        pred
    )

    disp.figure_.tight_layout()

    disp.figure_.savefig(
        REPORTS / "confusion_matrix.png",
        dpi=150
    )

    plt.close(disp.figure_)

    print(report)


if __name__ == "__main__":

    evaluate()