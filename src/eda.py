import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPORTS

from src.data_collection import load_data

from src.preprocessing import merge_and_clean

from src.feature_engineering import add_features


def create_eda():

    h, m = load_data()

    d = add_features(
        merge_and_clean(h, m)
    )

    REPORTS.mkdir(exist_ok=True)

    # Soil moisture distribution

    plt.figure(figsize=(7, 4))

    sns.histplot(
        data=d,
        x="soil_moisture",
        hue="disease_label",
        kde=True
    )

    plt.title(
        "Soil moisture by synthetic demo label"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS / "soil_moisture_distribution.png",
        dpi=150
    )

    plt.close()

    # Vegetation indices

    plt.figure(figsize=(7, 4))

    sns.scatterplot(
        data=d,
        x="ndvi",
        y="ndre",
        hue="disease_label",
        alpha=0.6
    )

    plt.title(
        "Vegetation indices - synthetic demo"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS / "vegetation_indices.png",
        dpi=150
    )

    plt.close()

    print("Charts saved in reports/")


if __name__ == "__main__":

    create_eda()