import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import io

# ----------------------------
# Swiss cities database
# ----------------------------
SWISS_CITIES = {
    "Zurich": (47.3769, 8.5417, 1),
    "Geneva": (46.2044, 6.1432, 1),
    "Basel": (47.5596, 7.5886, 1),
    "Bern": (46.948, 7.4474, 1),
    "Lausanne": (46.5197, 6.6323, 1),
    "Winterthur": (47.4988, 8.7237, 1),
    "Lucerne": (47.0502, 8.3093, 1),
    "St. Gallen": (47.4245, 9.3767, 1),
    "Lugano": (46.0037, 8.9511, 1),
    "Biel/Bienne": (47.1325, 7.2441, 1),
}

# ----------------------------
# Utility functions
# ----------------------------
def fetch_nasa_data(lat, lon, start_date, end_date):
    """Fetch hourly GHI and temperature data from NASA POWER."""
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"latitude={lat}&longitude={lon}"
        f"&start={start_date[:4]}0101&end={end_date[:4]}1231"
        f"&parameters=ALLSKY_SFC_SW_DWN,T2M&format=JSON"
    )
    res = requests.get(url)
    if res.status_code != 200:
        st.error("❌ NASA API request failed.")
        return None

    try:
        data = res.json()["properties"]["parameter"]
        ghi = pd.Series(data["ALLSKY_SFC_SW_DWN"])
        temp = pd.Series(data["T2M"])
        df = pd.DataFrame({"GHI": ghi, "Temp": temp})
        df.index = pd.date_range(
            start=f"{start_date[:4]}-01-01 00:00", periods=len(df), freq="H"
        )
        return df
    except Exception:
        st.error("⚠️ Unexpected NASA API response format.")
        return None


def simulate_pv_system(cfg):
    df = fetch_nasa_data(cfg["latitude"], cfg["longitude"], cfg["start_date"], cfg["end_date"])
    if df is None:
        return None, None

    area_total = cfg["num_cells"] * cfg["cell_area"]
    tilt_factor = max(0.1, abs((90 - cfg["slope"]) / 90))
    df["Power_W"] = df["GHI"] * area_total * cfg["efficiency"] * tilt_factor

    # Battery simulation
    soc = []
    battery_soc = cfg["battery_capacity"] / 2
    for p in df["Power_W"]:
        battery_soc += (p / 1000) / 1000 - (cfg["load_power"] / 1000) / 1000
        battery_soc = max(0, min(cfg["battery_capacity"], battery_soc))
        soc.append(battery_soc)
    df["SOC_kWh"] = soc

    daily = df.resample("D").sum()
    return df, daily


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="☀️ PV Simulator", layout="wide")

st.sidebar.title("🔧 System Configurator")

# City selection
city = st.sidebar.selectbox("Select City", list(SWISS_CITIES.keys()))
latitude, longitude, timezone = SWISS_CITIES[city]

# Config inputs
cfg = {
    "latitude": latitude,
    "longitude": longitude,
    "timezone": timezone,
    "slope": st.sidebar.slider("Tilt (°)", 0, 90, 30),
    "num_cells": st.sidebar.number_input("Number of Cells", 1, 500, 60),
    "cell_area": st.sidebar.number_input("Cell Area (m²)", 0.01, 2.0, 0.16),
    "efficiency": st.sidebar.number_input("PV Efficiency (0–1)", 0.01, 1.0, 0.2),
    "battery_capacity": st.sidebar.number_input("Battery Capacity (kWh)", 0.1, 100.0, 10.0),
    "load_power": st.sidebar.number_input("Constant Load (W)", 0, 5000, 2000),
    "start_date": st.sidebar.date_input("Start Date", datetime(2025, 1, 1)),
    "end_date": st.sidebar.date_input("End Date", datetime(2025, 1, 7)),
}

run = st.sidebar.button("🚀 Run Simulation")

# ----------------------------
# Main view
# ----------------------------
st.title("☀️ Photovoltaic (PV) System Simulator")

if run:
    with st.spinner("Fetching NASA data and running simulation..."):
        df, daily = simulate_pv_system({**cfg, "start_date": str(cfg["start_date"]), "end_date": str(cfg["end_date"])})
        if df is not None:
            st.success("✅ Simulation complete!")
            st.subheader(f"📍 Location: {city} ({latitude:.3f}, {longitude:.3f})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Average GHI (W/m²)", f"{df['GHI'].mean():.1f}")
            col2.metric("Avg Power Output (W)", f"{df['Power_W'].mean():.1f}")
            col3.metric("Battery SOC (kWh)", f"{df['SOC_kWh'].mean():.2f}")

            st.markdown("---")
            st.subheader("🔋 Hourly Battery State of Charge (SOC)")
            st.line_chart(df["SOC_kWh"])

            st.subheader("⚡ Hourly PV Power Output (W)")
            st.line_chart(df["Power_W"])

            st.subheader("☀️ Daily PV Energy Yield (Wh)")
            st.bar_chart(daily["Power_W"])

            # Export CSV
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=True)
            st.download_button(
                label="📥 Download Simulation Data as CSV",
                data=csv_buf.getvalue(),
                file_name="pv_simulation.csv",
                mime="text/csv",
            )
