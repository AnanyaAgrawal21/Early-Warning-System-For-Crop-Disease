import pandas as pd

KEYS = [
    "record_id",
    "plot_id",
    "timestamp"
]


def merge_and_clean(h, m):

    req_h = set(
        KEYS + [
            "blue",
            "green",
            "red",
            "red_edge",
            "nir",
            "disease_label"
        ]
    )

    req_m = set(
        KEYS + [
            "soil_moisture",
            "moisture_trend"
        ]
    )

    if req_h - set(h.columns):
        raise ValueError(
            f"Missing hyperspectral columns: "
            f"{req_h - set(h.columns)}"
        )

    if req_m - set(m.columns):
        raise ValueError(
            f"Missing moisture columns: "
            f"{req_m - set(m.columns)}"
        )

    h = h.copy()
    m = m.copy()

    for x in (h, m):

        x["timestamp"] = pd.to_datetime(
            x["timestamp"],
            errors="coerce"
        )

    d = h.merge(
        m,
        on=KEYS,
        how="inner",
        validate="one_to_one"
    )

    d = d.dropna(
        subset=["timestamp", "disease_label"]
    )

    cols = [
        "blue",
        "green",
        "red",
        "red_edge",
        "nir",
        "soil_moisture",
        "moisture_trend"
    ]

    for c in cols:

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

        d[c] = d[c].fillna(
            d[c].median()
        )

    d["disease_label"] = (
        d["disease_label"]
        .astype(int)
        .clip(0, 1)
    )

    if d.empty:

        raise ValueError(
            "No aligned records. Check record_id, "
            "plot_id and timestamp."
        )

    return d