import streamlit as st
import pandas as pd
import requests
import math
from datetime import datetime, timedelta
import plotly.graph_objects as go
from io import BytesIO

# ---------------------------
# Streamlit setup
# ---------------------------
st.set_page_config(page_title="Interactive PV Simulator", layout="wide")
st.title("☀️ Interactive Photovoltaic (PV) Simulator — NASA or Solargis")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("System Configuration")

data_source = st.sidebar.selectbox("Select Data Source", ["NASA POWER", "Solargis"])
latitude = st.sidebar.number_input("Latitude", value=46.9480)
longitude = st.sidebar.number_input("Longitude", value=7.4474)
timezone = st.sidebar.number_input("Timezone (UTC offset)", value=1)
slope = st.sidebar.number_input("Tilt (°)", value=30)
num_cells = st.sidebar.number_input("Number of Cells", value=60)
cell_area = st.sidebar.number_input("Cell Area (m² per cell)", value=0.16)
efficiency = st.sidebar.number_input("PV Efficiency (0–1)", value=0.20)
battery_capacity = st.sidebar.number_input("Battery Capacity (kWh)", value=10.0)
load_power = st.sidebar.number_input("Constant Load (W)", value=2000)

st.sidebar.markdown("---")
st.sidebar.header("Date Range")
default_end = datetime.utcnow()
default_start = default_end - timedelta(days=7)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

if data_source == "Solargis":
    solargis_key = st.sidebar.text_input("Solargis API Key", type="password")

# ---------------------------
# Data Retrieval
# ---------------------------
@st.cache_data
def get_nasa_data(lat, lon, start_date, end_date):
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_str,
        "end": end_str,
        "community": "RE",
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "format": "JSON",
        "user": "demo"
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()["properties"]["parameter"]
    df = pd.DataFrame({
        "GHI": pd.Series(data["ALLSKY_SFC_SW_DWN"]),
        "Temp": pd.Series(data["T2M"])
    })
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    return df

def get_solargis_data(lat, lon, start_date, end_date, api_key):
    """Template for Solargis API (replace URL with your plan endpoint)"""
    start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
    url = "https://solargis.info/api/series/time"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "lat": lat,
        "lon": lon,
        "attributes": "ghi,temp_air",
        "from": start_str,
        "to": end_str,
        "interval": "1h"
    }
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        raise ConnectionError(f"Solargis API request failed: {res.status_code}")
    data = res.json()
    times = [datetime.fromisoformat(entry["timestamp"].replace("Z", "")) for entry in data["data"]]
    ghi = [entry["ghi"] for entry in data["data"]]
    temp = [entry["temp_air"] for entry in data["data"]]
    df = pd.DataFrame({"GHI": ghi, "Temp": temp}, index=pd.to_datetime(times))
    return df

# ---------------------------
# PV & Battery Model
# ---------------------------
def solar_declination(day_of_year):
    return 23.45 * math.pi / 180 * math.sin(2 * math.pi * (284 + day_of_year) / 365)

def solar_hour_angle(hour, longitude, timezone_offset):
    solar_noon = 12 - (4 * longitude / 60) - timezone_offset
    return math.radians(15 * (hour - solar_noon))

def irradiance_on_tilted_surface(GHI, slope, latitude, decl, hour_angle):
    slope_rad = math.radians(slope)
    lat_rad = math.radians(latitude)
    cos_incidence = (
        math.sin(decl) * math.sin(lat_rad - slope_rad) +
        math.cos(decl) * math.cos(lat_rad - slope_rad) * math.cos(hour_angle)
    )
    cos_zenith = (
        math.sin(decl) * math.sin(lat_rad) +
        math.cos(decl) * math.cos(lat_rad) * math.cos(hour_angle)
    )
    return max(GHI * (cos_incidence / cos_zenith), 0) if cos_zenith > 0 else 0

def pv_power(latitude, longitude, timezone, slope, day_of_year, hour,
             ghi, temp_air, num_cells, cell_area, eff_ref=0.20, temp_coeff=-0.004, T_ref=25):
    decl = solar_declination(day_of_year)
    h_angle = solar_hour_angle(hour, longitude, timezone)
    irr_tilted = irradiance_on_tilted_surface(ghi, slope, latitude, decl, h_angle)
    total_area = num_cells * cell_area
    T_cell = temp_air + (irr_tilted / 800) * 20
    eff_temp = eff_ref * (1 + temp_coeff * (T_cell - T_ref))
    eff_temp = max(eff_temp, 0)
    return irr_tilted * total_area * eff_temp

def simulate_battery(power_series, load_power, capacity_kWh, dt_hours=1):
    capacity_Wh = capacity_kWh * 1000
    soc = 0.5 * capacity_Wh
    soc_series, net_series = [], []
    for pv_power_val in power_series:
        net = pv_power_val - load_power
        soc = min(capacity_Wh, max(0, soc + net * dt_hours))
        soc_series.append(soc / 1000)
        net_series.append(net)
    return pd.Series(soc_series, index=power_series.index), pd.Series(net_series, index=power_series.index)

# ---------------------------
# Run Simulation
# ---------------------------
if st.button("Run Simulation"):
    try:
        if data_source == "NASA POWER":
            df = get_nasa_data(latitude, longitude, start_date, end_date)
        else:
            if not solargis_key:
                st.error("Please enter your Solargis API key.")
                st.stop()
            df = get_solargis_data(latitude, longitude, start_date, end_date, solargis_key)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    df["hour"] = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    df["Power_W"] = [
        pv_power(latitude, longitude, timezone, slope,
                 day, hour, ghi, temp, num_cells, cell_area, efficiency)
        for day, hour, ghi, temp in zip(df["day_of_year"], df["hour"], df["GHI"], df["Temp"])
    ]

    df_daily = df.resample("D").sum(numeric_only=True) / 1000  # kWh/day
    soc, net_power = simulate_battery(df["Power_W"], load_power, battery_capacity)
    df["SOC_kWh"] = soc

    st.subheader("📊 Results Summary")
    st.metric("Average Daily PV Energy", f"{df_daily['Power_W'].mean():.2f} kWh/day")
    st.metric("Battery Capacity", f"{battery_capacity:.1f} kWh")
    st.metric("Load", f"{load_power:.0f} W")

    # ---------------------------
    # Plotly Interactive Charts
    # ---------------------------
    st.subheader("Interactive Visualization")

    tabs = st.tabs(["Hourly PV Power", "Daily Energy Yield", "Battery SOC"])

    with tabs[0]:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df.index, y=df["Power_W"], mode="lines", line=dict(color="orange"), name="PV Power"))
        fig1.update_layout(title="Hourly PV Power (W)", xaxis_title="Time", yaxis_title="Power [W]", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with tabs[1]:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_daily.index, y=df_daily["Power_W"], marker_color="skyblue"))
        fig2.update_layout(title="Daily PV Energy Yield (kWh/day)", xaxis_title="Date", yaxis_title="Energy [kWh]", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df["SOC_kWh"], mode="lines", line=dict(color="green"), name="Battery SOC"))
        fig3.update_layout(title="Battery State of Charge (kWh)", xaxis_title="Time", yaxis_title="SOC [kWh]", template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------
    # CSV Export
    st.subheader("📥 Export Data")
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer)
    csv_buffer.seek(0)
    st.download_button(
        label="Download PV & Battery Data (CSV)",
        data=csv_buffer,
        file_name=f"PV_Simulation_{data_source.replace(' ', '_')}.csv",
        mime="text/csv"
    )
