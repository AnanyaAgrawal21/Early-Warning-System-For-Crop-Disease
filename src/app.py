import streamlit as st
import matplotlib.pyplot as plt

from src.data_collection import (
    load_data,
    generate_sample_data
)

from src.preprocessing import merge_and_clean

from src.feature_engineering import add_features

from src.risk_prediction import (
    load_model,
    predict_risk
)

from src.config import MODEL


st.set_page_config(
    page_title="AgriNex Crop Disease Early Warning",
    page_icon="🌱",
    layout="wide"
)

st.title(
    "🌱 AgriNex — Crop Disease Early Warning System"
)

st.caption(
    "Hyperspectral imaging + soil-moisture sensing "
    "| Team Code: 2602379"
)

st.warning(
    "DEMO MODE: supplied sample data is synthetic, "
    "not field-validated, and not a disease diagnosis."
)


# Sidebar

with st.sidebar:

    st.header("Project Controls")

    if st.button("Regenerate synthetic demo data"):

        generate_sample_data()

        st.success(
            "Generated. Refresh the page."
        )


# Load data

try:

    h, m = load_data()

    clean = add_features(
        merge_and_clean(h, m)
    )

except Exception as e:

    st.error(str(e))
    st.stop()


# Check model

if not MODEL.exists():

    st.info(
        "Train model in terminal: "
        "python -m src.train_model"
    )

    st.stop()


# Predictions

try:

    result = predict_risk(
        clean,
        load_model()
    )

except Exception as e:

    st.error(str(e))
    st.stop()


# Dashboard metrics

a, b, c, d = st.columns(4)

a.metric(
    "Records",
    len(result)
)

b.metric(
    "High risk",
    int(
        (result.risk_level == "High").sum()
    )
)

c.metric(
    "Medium risk",
    int(
        (result.risk_level == "Medium").sum()
    )
)

d.metric(
    "Low risk",
    int(
        (result.risk_level == "Low").sum()
    )
)


# Results table

cols = [
    "record_id",
    "plot_id",
    "timestamp",
    "soil_moisture",
    "risk_probability",
    "risk_level",
    "recommended_action"
]

st.subheader(
    "Field-zone risk results"
)

st.dataframe(
    result[cols].sort_values(
        "risk_probability",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# Risk distribution chart

st.subheader(
    "Risk Distribution"
)

fig, ax = plt.subplots()

(
    result.risk_level
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )
    .plot(
        kind="bar",
        ax=ax
    )
)

ax.set_xlabel("Risk Level")
ax.set_ylabel("Records")

st.pyplot(fig)

plt.close(fig)


# Inspect a record

rid = st.selectbox(
    "Inspect a record",
    result.record_id.astype(str).tolist()
)

r = result[
    result.record_id.astype(str) == rid
].iloc[0]


x, y, z = st.columns(3)

x.metric(
    "Risk",
    r.risk_level
)

y.metric(
    "Risk score",
    f"{r.risk_probability:.1%}"
)

z.metric(
    "Soil moisture",
    f"{r.soil_moisture:.1f}"
)


st.write(
    "Suggested next step:",
    r.recommended_action
)


st.info(
    "Risk alerts prioritize inspection only. "
    "Farmer/agronomist verification is required "
    "before diagnosis or treatment."
)


# Download report

st.download_button(
    "Download risk report CSV",
    result[cols].to_csv(index=False).encode(),
    "crop_disease_risk_report.csv",
    "text/csv"
)