import numpy as np
import pandas as pd

from src.config import DATA, HYPER, MOISTURE


def generate_sample_data(n=1200, seed=42):

    # Synthetic demonstration data only
    rng = np.random.default_rng(seed)

    DATA.mkdir(exist_ok=True)

    ts = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="6h"
    ).astype(str)

    plot = rng.choice(
        [f"P{i:03}" for i in range(1, 31)],
        n
    )

    soil = np.clip(
        rng.normal(28, 9, n),
        5,
        60
    )

    trend = rng.normal(0, 2.5, n)

    blue = np.clip(
        rng.normal(.08, .018, n),
        .01, .2
    )

    green = np.clip(
        rng.normal(.12, .025, n),
        .02, .3
    )

    red = np.clip(
        rng.normal(.10, .025, n),
        .01, .3
    )

    red_edge = np.clip(
        rng.normal(.28, .06, n),
        .05, .6
    )

    nir = np.clip(
        rng.normal(.58, .10, n),
        .1, .9
    )

    p = np.clip(
        .08
        + .38 * ((soil < 18) | (soil > 48))
        + .38 * ((nir < .48) & (red > .105)),
        .02,
        .92
    )

    label = rng.binomial(1, p)

    ids = [
        f"R{i:05}"
        for i in range(n)
    ]

    h = pd.DataFrame({
        "record_id": ids,
        "plot_id": plot,
        "timestamp": ts,
        "blue": blue,
        "green": green,
        "red": red,
        "red_edge": red_edge,
        "nir": nir,
        "disease_label": label
    })

    m = pd.DataFrame({
        "record_id": ids,
        "plot_id": plot,
        "timestamp": ts,
        "soil_moisture": soil,
        "moisture_trend": trend
    })

    # Add missing values for cleaning demonstration
    for frame, col in [
        (h, "red_edge"),
        (m, "soil_moisture")
    ]:

        ix = rng.choice(
            n,
            max(1, n // 100),
            replace=False
        )

        frame.loc[ix, col] = np.nan

    h.to_csv(HYPER, index=False)
    m.to_csv(MOISTURE, index=False)

    return h, m


def load_data():

    if not HYPER.exists() or not MOISTURE.exists():
        return generate_sample_data()

    return (
        pd.read_csv(HYPER),
        pd.read_csv(MOISTURE)
    )


if __name__ == "__main__":

    generate_sample_data()

    print(
        "Synthetic demo CSV files created in data/."
    )