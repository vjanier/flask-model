import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Airline Price Predictor", page_icon="✈️")


@st.cache_resource
def load():
    model = joblib.load("airline_price_model.joblib")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    with open("flight_rules.json") as f:
        rules = json.load(f)  # rules[stops][departure][arrival] = [min, median, max] hours
    return model, meta, rules


model, meta, rules = load()
cats = meta["categories"]

st.title("✈️ Airline Price Predictor")

c1, c2 = st.columns(2)
airline = c1.selectbox("Airline", cats["airline"])
# Only Vistara and Air_India sell Business; every other airline is Economy only.
BUSINESS_AIRLINES = {"Vistara", "Air_India"}
class_options = cats["class"] if airline in BUSINESS_AIRLINES else ["Economy"]
flight_class = c2.selectbox("Class", class_options, key=f"class|{airline}")
if airline not in BUSINESS_AIRLINES:
    c2.caption(f"{airline} only offers Economy.")
source = c1.selectbox("From", cats["source_city"])
dest = c2.selectbox("To", cats["destination_city"])

# Stops -> departure -> arrival -> duration: each choice limits the next one.
stops = st.selectbox("Stops", [s for s in cats["stops"] if s in rules])
dep_options = [t for t in cats["departure_time"] if t in rules[stops]]
dep = c1.selectbox("Departure time", dep_options)
arr_options = [t for t in cats["arrival_time"] if t in rules[stops][dep]]
arr = c2.selectbox("Arrival time", arr_options)

lo, mid, hi = rules[stops][dep][arr]
key = f"{stops}|{dep}|{arr}"  # new key = widget resets when the combination changes
duration = st.number_input(
    f"Duration (hours), typical range for this route: {lo} - {hi}",
    min_value=float(lo), max_value=float(hi), value=float(mid), step=0.25, key=f"dur|{key}",
)
days_left = st.slider("Days until departure", *meta["days_left_range"], value=15)

if st.button("Predict price", type="primary"):
    if source == dest:
        st.warning("From and To are the same city.")
    else:
        row = pd.DataFrame([{
            "airline": airline, "source_city": source, "departure_time": dep,
            "stops": stops, "arrival_time": arr, "destination_city": dest,
            "class": flight_class, "duration": duration, "days_left": days_left,
        }])[meta["feature_cols"]]
        price = model.predict(row)[0]
        st.metric("Estimated price", f"{price:,.0f} {meta['currency']}")
