"""Airline price prediction - Streamlit app.

Run locally:  streamlit run streamlit_app.py
Needs airline_price_model.joblib and model_metadata.json in the same folder
(or set MODEL_PATH / META_PATH env vars).
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", "airline_price_model.joblib")
META_PATH = os.getenv("META_PATH", "model_metadata.json")

st.set_page_config(page_title="Airline Price Predictor", page_icon="✈️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, MODEL_PATH))


@st.cache_data
def load_meta():
    with open(os.path.join(BASE_DIR, META_PATH)) as f:
        return json.load(f)


def validate(df: pd.DataFrame, meta: dict) -> list:
    """Return a list of error strings (empty if the data is valid)."""
    errors = []
    for col, allowed in meta["categories"].items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue
        bad = sorted(set(df[col].astype(str)) - set(allowed))
        if bad:
            errors.append(f"{col}: invalid values {bad}. Allowed: {allowed}")
    lo, hi = meta["duration_range"]
    if "duration" in df.columns and not df["duration"].between(lo, hi).all():
        errors.append(f"duration must be between {lo} and {hi}")
    lo, hi = meta["days_left_range"]
    if "days_left" in df.columns and not df["days_left"].between(lo, hi).all():
        errors.append(f"days_left must be between {lo} and {hi}")
    for col in meta["feature_cols"]:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    return errors


try:
    model = load_model()
    meta = load_meta()
except Exception as e:  # missing files, version mismatch, etc.
    st.error(f"Could not load model files: {e}")
    st.stop()

cats = meta["categories"]
currency = meta.get("currency", "")

st.title("✈️ Airline Price Predictor")
st.caption("Estimate a flight ticket price from its details.")

tab_single, tab_batch, tab_about = st.tabs(["Single prediction", "Batch (CSV)", "Model info"])

# ---------------------------------------------------------------- single
with tab_single:
    with st.form("flight_form"):
        c1, c2 = st.columns(2)
        airline = c1.selectbox("Airline", cats["airline"])
        flight_class = c2.selectbox("Class", cats["class"])

        c1, c2 = st.columns(2)
        source_city = c1.selectbox("From", cats["source_city"])
        destination_city = c2.selectbox("To", cats["destination_city"])

        c1, c2 = st.columns(2)
        departure_time = c1.selectbox("Departure time", cats["departure_time"])
        arrival_time = c2.selectbox("Arrival time", cats["arrival_time"])

        stops = st.selectbox("Stops", cats["stops"])

        d_lo, d_hi = meta["duration_range"]
        l_lo, l_hi = meta["days_left_range"]
        c1, c2 = st.columns(2)
        duration = c1.number_input(
            "Duration (hours)", min_value=float(d_lo), max_value=float(d_hi),
            value=min(max(2.0, float(d_lo)), float(d_hi)), step=0.25,
        )
        days_left = c2.number_input(
            "Days until departure", min_value=int(l_lo), max_value=int(l_hi),
            value=min(max(15, int(l_lo)), int(l_hi)), step=1,
        )

        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        if source_city == destination_city:
            st.warning("Source and destination cities are the same.")
        row = {
            "airline": airline,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": flight_class,
            "duration": float(duration),
            "days_left": int(days_left),
        }
        df = pd.DataFrame([row])
        errors = validate(df, meta)
        if errors:
            for err in errors:
                st.error(err)
        else:
            price = float(model.predict(df[meta["feature_cols"]])[0])
            st.metric("Estimated price", f"{price:,.2f} {currency}")

# ----------------------------------------------------------------- batch
with tab_batch:
    st.write(
        "Upload a CSV with these columns: "
        + ", ".join(f"`{c}`" for c in meta["feature_cols"])
    )
    template = pd.DataFrame(columns=meta["feature_cols"])
    st.download_button(
        "Download empty template", template.to_csv(index=False),
        file_name="flights_template.csv", mime="text/csv",
    )
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        if len(data) == 0 or len(data) > 1000:
            st.error("Upload between 1 and 1000 rows.")
        else:
            errors = validate(data, meta)
            if errors:
                for err in errors:
                    st.error(err)
            else:
                out = data.copy()
                out["predicted_price"] = model.predict(data[meta["feature_cols"]]).round(2)
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Download predictions", out.to_csv(index=False),
                    file_name="predictions.csv", mime="text/csv",
                )

# ----------------------------------------------------------------- about
with tab_about:
    st.write(f"**Trained at:** {meta.get('trained_at', 'n/a')}")
    st.write(f"**scikit-learn version:** {meta.get('sklearn_version', 'n/a')}")
    st.write("**Test scores:**")
    st.json(meta.get("test_scores", {}))
