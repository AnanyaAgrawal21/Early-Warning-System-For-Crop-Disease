import numpy as np

FEATURES = [
    "blue",
    "green",
    "red",
    "red_edge",
    "nir",
    "soil_moisture",
    "moisture_trend",
    "ndvi",
    "ndre",
    "gndvi"
]


def add_features(df):

    d = df.copy()

    e = 1e-8

    # Normalized Difference Vegetation Index
    d["ndvi"] = (
        (d.nir - d.red) /
        (d.nir + d.red + e)
    )

    # Normalized Difference Red Edge
    d["ndre"] = (
        (d.nir - d.red_edge) /
        (d.nir + d.red_edge + e)
    )

    # Green Normalized Difference Vegetation Index
    d["gndvi"] = (
        (d.nir - d.green) /
        (d.nir + d.green + e)
    )

    d[FEATURES] = d[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    d[FEATURES] = d[FEATURES].fillna(
        d[FEATURES].median()
    )

    return d