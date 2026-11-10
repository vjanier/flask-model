import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Airline Price Predictor", page_icon="✈️")


@st.cache_resource
def load():
    model = joblib.load("airline_price_model.joblib")
    with open("model_metadata.json") as f:
        return model, json.load(f)


model, meta = load()
cats = meta["categories"]

st.title("✈️ Airline Price Predictor")

c1, c2 = st.columns(2)
airline = c1.selectbox("Airline", cats["airline"])
flight_class = c2.selectbox("Class", cats["class"])
source = c1.selectbox("From", cats["source_city"])
dest = c2.selectbox("To", cats["destination_city"])
dep = c1.selectbox("Departure time", cats["departure_time"])
arr = c2.selectbox("Arrival time", cats["arrival_time"])
stops = c1.selectbox("Stops", cats["stops"])
duration = c2.number_input("Duration (hours)", *meta["duration_range"], value=2.0, step=0.25)
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
