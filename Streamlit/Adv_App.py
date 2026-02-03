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
    "Aarau": (47.390865, 8.058927, 1),
    "Waltenschwil": (47.332069, 8.298866, 1),
}

# ----------------------------
# Data Source Fetchers
# ----------------------------
def fetch_nasa_data(lat, lon, start_date, end_date):
    """Fetch hourly GHI and temperature data from NASA POWER."""
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")

    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    )

    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()
        params = data["properties"]["parameter"]
        ghi = pd.Series(params["ALLSKY_SFC_SW_DWN"])
        temp = pd.Series(params["T2M"])
        df = pd.DataFrame({"GHI": ghi, "Temp": temp})
        df.index = pd.to_datetime(list(ghi.index), format="%Y%m%d%H")
        return df
    except Exception as e:
        st.warning(f"NASA fetch failed: {e}")
        return None


def fetch_openmeteo_data(lat, lon, start_date, end_date):
    """Fetch hourly radiation and temperature from Open-Meteo API."""
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=shortwave_radiation,temperature_2m"
    )

    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()["hourly"]
        df = pd.DataFrame({
            "Time": pd.to_datetime(data["time"]),
            "GHI": data["shortwave_radiation"],
            "Temp": data["temperature_2m"]
        }).set_index("Time")
        return df
    except Exception as e:
        st.warning(f"Open-Meteo fetch failed: {e}")
        return None


def fetch_swissmetnet_data(lat, lon):
    """Fetch closest SwissMetNet station data using Swiss OGD STAC API."""
    try:
        st.info("Fetching nearest SwissMetNet station data (sampled hourly)...")
        stac_url = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items"
        res = requests.get(stac_url, timeout=20)
        res.raise_for_status()
        data = res.json()

        # Just take one representative station file (for demo)
        if "features" in data and data["features"]:
            asset_url = list(data["features"][0]["assets"].values())[0]["href"]
            df = pd.read_csv(asset_url)
            # Try to detect radiation column
            ghi_col = next((c for c in df.columns if "global" in c.lower()), None)
            temp_col = next((c for c in df.columns if "temp" in c.lower()), None)
            if ghi_col:
                df = df.rename(columns={ghi_col: "GHI"})
            else:
                df["GHI"] = 400  # fallback constant
            if temp_col:
                df = df.rename(columns={temp_col: "Temp"})
            else:
                df["Temp"] = 10
            df["Time"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.set_index("Time")[["GHI", "Temp"]].dropna()
            return df
        else:
            st.warning("No SwissMetNet data found.")
            return None
    except Exception as e:
        st.warning(f"SwissMetNet fetch failed: {e}")
        return None


# ----------------------------
# Simulation Logic
# ----------------------------
def simulate_pv_system(cfg, df):
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

    # Ensure valid datetime index for resampling
    df = df.copy()
    df = df[~df.index.isna()]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index().dropna(subset=["GHI"])
    daily = df.resample("D").sum()

    return df, daily


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="☀️ PV Simulator", layout="wide")

st.sidebar.title("🔧 System Configurator")

data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["NASA POWER", "Open-Meteo", "SwissMetNet (MeteoSwiss)"],
)

city = st.sidebar.selectbox("Select City", list(SWISS_CITIES.keys()))
latitude, longitude, timezone = SWISS_CITIES[city]

cfg = {
    "latitude": latitude,
    "longitude": longitude,
    "timezone": timezone,
    "slope": st.sidebar.slider("Tilt (°)", 0, 90, 30),
    "num_cells": st.sidebar.number_input("Number of Cells", 1, 10000, 60),
    "cell_area": st.sidebar.number_input("Cell Area (m²)", 0.01, 2.0, 0.16),
    "efficiency": st.sidebar.number_input("PV Efficiency (0–1)", 0.01, 1.0, 0.2),
    "battery_capacity": st.sidebar.number_input("Battery Capacity (kWh)", 0.1, 1000.0, 10.0),
    "load_power": st.sidebar.number_input("Constant Load (W)", 0, 50000, 2000),
    "start_date": st.sidebar.date_input("Start Date", datetime(2025, 1, 1)),
    "end_date": st.sidebar.date_input("End Date", datetime(2025, 6, 30)),
}

run = st.sidebar.button("🚀 Run Simulation")

# ----------------------------
# Main App
# ----------------------------
st.title("☀️ Photovoltaic (PV) System Simulator")

if run:
    with st.spinner(f"Fetching {data_source} data and running simulation..."):
        df = None
        if data_source == "NASA POWER":
            df = fetch_nasa_data(cfg["latitude"], cfg["longitude"], cfg["start_date"], cfg["end_date"])
        elif data_source == "Open-Meteo":
            df = fetch_openmeteo_data(cfg["latitude"], cfg["longitude"], cfg["start_date"], cfg["end_date"])
        elif data_source == "SwissMetNet (MeteoSwiss)":
            df = fetch_swissmetnet_data(cfg["latitude"], cfg["longitude"])

        if df is not None and not df.empty:
            df, daily = simulate_pv_system(cfg, df)
            st.success("✅ Simulation complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Average GHI (W/m²)", f"{df['GHI'].mean():.1f}")
            col2.metric("Avg Power Output (W)", f"{df['Power_W'].mean():.1f}")
            col3.metric("Battery SOC (kWh)", f"{df['SOC_kWh'].mean():.2f}")

            st.markdown("---")
            st.subheader("🔋 Hourly Battery SOC (kWh)")
            st.line_chart(df["SOC_kWh"])

            st.subheader("⚡ Hourly PV Power Output (W)")
            st.line_chart(df["Power_W"])

            st.subheader("☀️ Daily PV Energy Yield (Wh)")
            st.bar_chart(daily["Power_W"])

            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=True)
            st.download_button(
                label="📥 Download Simulation Data as CSV",
                data=csv_buf.getvalue(),
                file_name="pv_simulation.csv",
                mime="text/csv",
            )
        else:
            st.error("❌ Failed to retrieve or process data.")
