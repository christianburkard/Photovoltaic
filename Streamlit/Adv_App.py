"""
Photovoltaic (PV) System Simulator using NASA POWER API
========================================================

This application simulates a residential solar PV system with battery storage,
using real hourly irradiance data from NASA POWER API.

Key Terms:
- GHI: Global Horizontal Irradiance (W/m²) - solar energy hitting a horizontal surface
- SOC: State of Charge (kWh) - current battery energy level
- PV: Photovoltaic - converts sunlight to electricity

Requirements:
    pip install streamlit pandas numpy requests plotly kaleido

Note: kaleido is required for PNG export functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ----------------------------
# Constants
# ----------------------------
# Battery efficiency factors
CHARGE_EFFICIENCY = 0.95  # 95% efficiency when charging
DISCHARGE_EFFICIENCY = 0.95  # 95% efficiency when discharging

# Orientation adjustment factors (empirically derived for Swiss latitudes)
ORIENTATION_WEIGHTS = {
    "North": 0.4,  # North-facing receives least direct sun
    "South": 1.0,  # South-facing optimal in northern hemisphere
    "East": 0.75,  # Morning sun
    "West": 0.75,  # Afternoon sun
}

# Load profile time periods (hours)
LOAD_NIGHT = (0, 6)  # Midnight to 6 AM
LOAD_MORNING = (6, 9)  # Morning peak
LOAD_DAY = (9, 17)  # Daytime baseline
LOAD_EVENING = (17, 21)  # Evening peak
LOAD_LATE = (21, 24)  # Late evening

# Load multipliers relative to average
LOAD_FACTORS = {
    "night": 0.3,
    "morning": 0.8,
    "day": 0.5,
    "evening": 1.0,
    "late": 0.6,
}

# API configuration
NASA_API_TIMEOUT = 30  # seconds
NASA_API_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Date limits
MIN_DATE = datetime(1984, 1, 1)  # NASA POWER data availability
MAX_DATE = datetime.now() - timedelta(days=2)  # Recent data may be incomplete

# ----------------------------
# Swiss Cities Database
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
# Plot Creation Helper
# ----------------------------
def create_plot_with_download(fig: go.Figure, filename: str, title: str) -> None:
    """
    Display a Plotly figure with a download button for PNG export.
    
    Args:
        fig: Plotly figure object
        filename: Base filename for the download (without extension)
        title: Display title for the section
    """
    st.plotly_chart(fig, width='stretch')
    
    # Create download button for PNG
    img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
    st.download_button(
        label=f"💾 Save as PNG",
        data=img_bytes,
        file_name=f"{filename}.png",
        mime="image/png",
        key=f"download_{filename}",
        width='content'
    )


def create_power_flow_plot(df: pd.DataFrame) -> go.Figure:
    """Create power flow visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Power_Total"],
        mode='lines',
        name='PV Power',
        line=dict(color='#FDB462', width=2),
        fill='tozeroy',
        fillcolor='rgba(253, 180, 98, 0.3)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Load_W"],
        mode='lines',
        name='Load',
        line=dict(color='#E41A1C', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Net_Power"],
        mode='lines',
        name='Net Power',
        line=dict(color='#377EB8', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Power Flow: PV Generation vs Household Load",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    return fig


def create_orientation_plot(df: pd.DataFrame) -> go.Figure:
    """Create orientation comparison plot."""
    fig = go.Figure()
    
    colors = {
        'North': '#984EA3',
        'South': '#FF7F00',
        'East': '#4DAF4A',
        'West': '#377EB8'
    }
    
    for orientation in ['North', 'South', 'East', 'West']:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f"Power_{orientation}"],
            mode='lines',
            name=f'{orientation}-facing',
            line=dict(color=colors[orientation], width=2),
            stackgroup='one'
        ))
    
    fig.update_layout(
        title="Power Output by Panel Orientation",
        xaxis_title="Time",
        yaxis_title="Power (W)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    return fig


def create_battery_plot(df: pd.DataFrame, capacity: float) -> go.Figure:
    """Create battery state of charge plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["SOC_kWh"],
        mode='lines',
        name='Battery SOC',
        line=dict(color='#4DAF4A', width=3),
        fill='tozeroy',
        fillcolor='rgba(77, 175, 74, 0.3)'
    ))
    
    # Add capacity reference line
    fig.add_hline(
        y=capacity,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max Capacity ({capacity} kWh)",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Empty",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Battery State of Charge (SOC)",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )
    
    return fig


def create_daily_energy_plot(daily: pd.DataFrame) -> go.Figure:
    """Create daily energy production bar chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily.index,
        y=daily["Power_Total"],
        name='PV Generation',
        marker_color='#FDB462',
        text=daily["Power_Total"].round(1),
        textposition='outside',
        texttemplate='%{text} kWh'
    ))
    
    fig.add_trace(go.Bar(
        x=daily.index,
        y=daily["Load_W"],
        name='Load Consumption',
        marker_color='#E41A1C',
        text=daily["Load_W"].round(1),
        textposition='outside',
        texttemplate='%{text} kWh'
    ))
    
    fig.update_layout(
        title="Daily Energy Production vs Consumption",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        barmode='group',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    return fig


def create_self_sufficiency_plot(daily: pd.DataFrame) -> go.Figure:
    """Create daily self-sufficiency chart."""
    fig = go.Figure()
    
    colors = ['#4DAF4A' if x >= 75 else '#FDB462' if x >= 50 else '#E41A1C' 
              for x in daily["Self_Sufficiency_%"]]
    
    fig.add_trace(go.Bar(
        x=daily.index,
        y=daily["Self_Sufficiency_%"],
        name='Self-Sufficiency',
        marker_color=colors,
        text=daily["Self_Sufficiency_%"].round(1),
        textposition='outside',
        texttemplate='%{text}%'
    ))
    
    # Add reference lines
    fig.add_hline(y=100, line_dash="dash", line_color="green", 
                  annotation_text="100% Self-Sufficient", annotation_position="right")
    fig.add_hline(y=75, line_dash="dot", line_color="orange", 
                  annotation_text="75%", annotation_position="right")
    fig.add_hline(y=50, line_dash="dot", line_color="red", 
                  annotation_text="50%", annotation_position="right")
    
    fig.update_layout(
        title="Daily Self-Sufficiency Ratio",
        xaxis_title="Date",
        yaxis_title="Self-Sufficiency (%)",
        hovermode='x',
        template="plotly_white",
        height=500,
        yaxis=dict(range=[0, min(110, daily["Self_Sufficiency_%"].max() + 10)])
    )
    
    return fig


def create_grid_interaction_plot(daily: pd.DataFrame) -> go.Figure:
    """Create grid import/export visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily.index,
        y=-daily["Grid_Import_W"],  # Negative for import
        name='Grid Import',
        marker_color='#E41A1C',
        text=daily["Grid_Import_W"].round(1),
        textposition='outside',
        texttemplate='%{text} kWh'
    ))
    
    fig.add_trace(go.Bar(
        x=daily.index,
        y=daily["Grid_Export_W"],
        name='Grid Export',
        marker_color='#4DAF4A',
        text=daily["Grid_Export_W"].round(1),
        textposition='outside',
        texttemplate='%{text} kWh'
    ))
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    
    fig.update_layout(
        title="Daily Grid Interaction (Import vs Export)",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500
    )
    
    return fig


# ----------------------------
# Input Validation
# ----------------------------
class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_inputs(cfg: Dict) -> None:
    """
    Validate all configuration inputs before running simulation.
    
    Args:
        cfg: Configuration dictionary with all system parameters
        
    Raises:
        ValidationError: If any validation check fails
    """
    # Date validation
    if cfg["start_date"] > cfg["end_date"]:
        raise ValidationError("Start date must be before end date")
    
    if cfg["start_date"] < MIN_DATE.date():
        raise ValidationError(f"Start date must be after {MIN_DATE.date()}")
    
    if cfg["end_date"] > MAX_DATE.date():
        raise ValidationError(f"End date must be before {MAX_DATE.date()} (recent data incomplete)")
    
    date_range = (cfg["end_date"] - cfg["start_date"]).days
    if date_range > 365:
        raise ValidationError("Date range cannot exceed 365 days")
    
    if date_range < 1:
        raise ValidationError("Date range must be at least 1 day")
    
    # Physical parameter validation
    if cfg["slope"] < 0 or cfg["slope"] > 90:
        raise ValidationError("Tilt must be between 0° and 90°")
    
    if cfg["num_cells"] <= 0:
        raise ValidationError("Number of cells must be positive")
    
    if cfg["cell_area"] <= 0:
        raise ValidationError("Cell area must be positive")
    
    if cfg["efficiency"] <= 0 or cfg["efficiency"] > 1:
        raise ValidationError("PV efficiency must be between 0 and 1")
    
    if cfg["battery_capacity"] <= 0:
        raise ValidationError("Battery capacity must be positive")
    
    if cfg["avg_load"] <= 0:
        raise ValidationError("Average load must be positive")
    
    # Sanity checks
    total_area = cfg["num_cells"] * cfg["cell_area"]
    if total_area > 1000:
        raise ValidationError(f"Total PV area ({total_area:.1f} m²) seems unrealistic for residential")
    
    if cfg["avg_load"] > 10000:
        raise ValidationError("Average load > 10 kW seems high for residential")


# ----------------------------
# NASA POWER API with Caching
# ----------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_nasa_data(
    lat: float, 
    lon: float, 
    start_date: datetime.date, 
    end_date: datetime.date
) -> Optional[pd.DataFrame]:
    """
    Fetch hourly GHI and temperature data from NASA POWER API.
    
    The NASA POWER API provides satellite-derived solar irradiance data
    with global coverage at hourly resolution.
    
    Parameters:
        - ALLSKY_SFC_SW_DWN: All-sky surface shortwave downward irradiance (GHI)
        - T2M: Temperature at 2 meters above surface
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        DataFrame with datetime index and columns: GHI (W/m²), Temp (°C)
        Returns None if request fails
    """
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")
    
    url = (
        f"{NASA_API_BASE_URL}?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    )

    try:
        with st.spinner("📡 Fetching data from NASA POWER API..."):
            res = requests.get(url, timeout=NASA_API_TIMEOUT)
            res.raise_for_status()
            
        data = res.json()
        
        # Check if data exists
        if "properties" not in data or "parameter" not in data["properties"]:
            st.error("❌ Invalid response from NASA API")
            return None
        
        params = data["properties"]["parameter"]
        
        # Extract time series data
        ghi = pd.Series(params.get("ALLSKY_SFC_SW_DWN", {}))
        temp = pd.Series(params.get("T2M", {}))
        
        if ghi.empty:
            st.error("❌ No GHI data returned from NASA API")
            return None
        
        # Create DataFrame
        df = pd.DataFrame({"GHI": ghi, "Temp": temp})
        df.index = pd.to_datetime(list(ghi.index), format="%Y%m%d%H", errors="coerce")
        
        # Clean data
        df = df.dropna(subset=["GHI"]).sort_index()
        
        # Replace negative GHI values (nighttime) with 0
        df.loc[df["GHI"] < 0, "GHI"] = 0
        
        if df.empty:
            st.error("❌ No valid data after cleaning")
            return None
            
        st.success(f"✅ Retrieved {len(df)} hours of data")
        return df
        
    except requests.Timeout:
        st.error(f"⏱️ Request timed out after {NASA_API_TIMEOUT} seconds. Try again later.")
        return None
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            st.error("❌ Data not found for specified location/dates")
        elif e.response.status_code == 429:
            st.error("❌ Too many requests. Please wait a moment and try again.")
        else:
            st.error(f"❌ HTTP Error {e.response.status_code}: {e}")
        return None
    except requests.ConnectionError:
        st.error("❌ Connection error. Check your internet connection.")
        return None
    except ValueError as e:
        st.error(f"❌ Data parsing error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


# ----------------------------
# Orientation Factors (Improved Physics)
# ----------------------------
def compute_orientation_factors(hours: np.ndarray, latitude: float) -> Dict[str, np.ndarray]:
    """
    Compute irradiance multipliers for different panel orientations.
    
    Uses simplified cosine model adjusted for latitude. More sophisticated
    models would include solar declination and azimuth angles.
    
    Args:
        hours: Array of hours (0-23)
        latitude: Location latitude for seasonal adjustment
        
    Returns:
        Dictionary mapping orientation to irradiance factor arrays
    """
    # Hour angle from solar noon (degrees)
    # Solar noon is approximately at 12:00 (simplified)
    hour_angle = (hours - 12) * 15  # 15 degrees per hour
    
    # Latitude adjustment factor (higher latitude = more seasonal variation)
    lat_factor = np.abs(latitude) / 90.0
    
    factors = {}
    
    # North-facing: Minimal direct sun (mainly diffuse)
    factors["North"] = np.full_like(hours, ORIENTATION_WEIGHTS["North"], dtype=float)
    
    # South-facing: Optimal for northern hemisphere
    # Peak at solar noon, cosine falloff
    factors["South"] = ORIENTATION_WEIGHTS["South"] * np.maximum(
        0.1, 
        np.cos(np.radians(hour_angle * (0.8 + 0.4 * lat_factor)))
    )
    
    # East-facing: Peak in morning (6-12)
    morning_peak_angle = (hours - 9) * 12  # Peak at 9 AM
    factors["East"] = ORIENTATION_WEIGHTS["East"] * np.maximum(
        0.1,
        np.cos(np.radians(morning_peak_angle))
    )
    
    # West-facing: Peak in afternoon (12-18)
    afternoon_peak_angle = (hours - 15) * 12  # Peak at 3 PM
    factors["West"] = ORIENTATION_WEIGHTS["West"] * np.maximum(
        0.1,
        np.cos(np.radians(afternoon_peak_angle))
    )
    
    return factors


# ----------------------------
# Tilt Factor (Improved Model)
# ----------------------------
def compute_tilt_factor(slope: float, latitude: float) -> float:
    """
    Compute annual average tilt factor for angled panels.
    
    Optimal tilt angle is approximately equal to latitude for year-round
    performance. This function computes a correction factor.
    
    Args:
        slope: Panel tilt angle in degrees (0 = horizontal, 90 = vertical)
        latitude: Location latitude
        
    Returns:
        Multiplicative factor (0-1) for irradiance collection
    """
    optimal_tilt = np.abs(latitude)
    
    # Deviation from optimal
    tilt_deviation = np.abs(slope - optimal_tilt)
    
    # Cosine loss model with smooth falloff
    # At optimal tilt: factor = 1.0
    # At 0° or 90°: factor decreases based on latitude
    factor = np.cos(np.radians(tilt_deviation * 0.7))
    
    # Ensure reasonable minimum (horizontal panels still work)
    return max(0.5, factor)


# ----------------------------
# Load Profile (Improved)
# ----------------------------
def generate_load_profile(df: pd.DataFrame, avg_load: float) -> pd.DataFrame:
    """
    Generate realistic hourly household electricity load profile.
    
    Based on typical residential consumption patterns:
    - Low overnight (sleeping)
    - Morning peak (breakfast, getting ready)
    - Moderate daytime (background loads)
    - Evening peak (cooking, lighting, appliances)
    - Late evening decline (winding down)
    
    Args:
        df: DataFrame with datetime index
        avg_load: Average household load in Watts
        
    Returns:
        DataFrame with added 'Load_W' column
    """
    hours = df.index.hour
    
    # Assign load factors based on time of day
    base = np.select(
        [
            (hours >= LOAD_NIGHT[0]) & (hours < LOAD_NIGHT[1]),
            (hours >= LOAD_MORNING[0]) & (hours < LOAD_MORNING[1]),
            (hours >= LOAD_DAY[0]) & (hours < LOAD_DAY[1]),
            (hours >= LOAD_EVENING[0]) & (hours < LOAD_EVENING[1]),
            (hours >= LOAD_LATE[0]) | (hours < LOAD_NIGHT[0]),
        ],
        [
            LOAD_FACTORS["night"],
            LOAD_FACTORS["morning"],
            LOAD_FACTORS["day"],
            LOAD_FACTORS["evening"],
            LOAD_FACTORS["late"],
        ],
        default=LOAD_FACTORS["day"]
    )
    
    # Add realistic random variation (±10%)
    np.random.seed(42)  # Reproducible results
    noise = np.random.uniform(0.9, 1.1, len(df))
    
    df["Load_W"] = avg_load * base * noise
    
    return df


# ----------------------------
# PV System Simulation (Vectorized)
# ----------------------------
def simulate_pv_system(cfg: Dict, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate complete PV system with battery storage.
    
    Physics model:
    1. Compute irradiance for each panel orientation
    2. Convert irradiance to DC power using PV efficiency
    3. Apply tilt correction factor
    4. Generate household load profile
    5. Compute net power (generation - consumption)
    6. Simulate battery charging/discharging with efficiency losses
    
    Args:
        cfg: System configuration dictionary
        df: DataFrame with GHI data
        
    Returns:
        Tuple of (hourly_data, daily_summary)
    """
    # Total PV area
    area_total = cfg["num_cells"] * cfg["cell_area"]
    
    # Tilt correction (improved physics model)
    tilt_factor = compute_tilt_factor(cfg["slope"], cfg["latitude"])
    
    hours = df.index.hour
    
    # Compute orientation-weighted irradiance and power
    orientation_factors = compute_orientation_factors(hours, cfg["latitude"])
    
    for orientation, factor in orientation_factors.items():
        # Power = GHI × orientation_factor × area × efficiency × tilt_factor
        # Units: (W/m²) × (m²) × (efficiency) = W
        df[f"Power_{orientation}"] = (
            df["GHI"] * 
            factor * 
            area_total * 
            cfg["efficiency"] * 
            tilt_factor
        )
    
    # Total PV power output (sum of all orientations)
    df["Power_Total"] = df[[f"Power_{orientation}" for orientation in orientation_factors]].sum(axis=1)
    
    # Generate load profile
    df = generate_load_profile(df, cfg["avg_load"])
    
    # Net power (positive = excess, negative = deficit)
    df["Net_Power"] = df["Power_Total"] - df["Load_W"]
    
    # ----------------------------
    # Battery State of Charge (SOC)
    # ----------------------------
    # Energy change per hour (W → kWh)
    delta_E = df["Net_Power"].values / 1000.0
    
    # Apply efficiency losses
    charging = delta_E > 0
    discharging = delta_E < 0
    
    delta_E[charging] *= CHARGE_EFFICIENCY  # Loss when charging
    delta_E[discharging] /= DISCHARGE_EFFICIENCY  # Loss when discharging
    
    # Cumulative energy (starting from mid-capacity)
    soc = np.cumsum(delta_E)
    
    # Initialize at 50% capacity
    initial_soc = cfg["battery_capacity"] / 2
    soc = soc - soc[0] + initial_soc
    
    # Clip to battery limits
    soc = np.clip(soc, 0, cfg["battery_capacity"])
    
    df["SOC_kWh"] = soc
    
    # Compute grid interaction
    # Positive = export to grid, Negative = import from grid
    soc_diff = np.diff(soc, prepend=soc[0])
    df["Grid_Power_W"] = df["Net_Power"] - (soc_diff * 1000)  # kWh → W
    
    # Calculate self-consumption metrics
    df["Self_Consumed_W"] = np.minimum(df["Power_Total"], df["Load_W"])
    df["Grid_Import_W"] = np.maximum(0, -df["Grid_Power_W"])
    df["Grid_Export_W"] = np.maximum(0, df["Grid_Power_W"])
    
    # Daily aggregates
    daily = df.resample("D").agg({
        "Power_Total": "sum",
        "Load_W": "sum",
        "Self_Consumed_W": "sum",
        "Grid_Import_W": "sum",
        "Grid_Export_W": "sum",
        "GHI": "mean",
        "Temp": "mean",
    })
    
    # Convert W to kWh for daily values (sum of hourly W values)
    for col in ["Power_Total", "Load_W", "Self_Consumed_W", "Grid_Import_W", "Grid_Export_W"]:
        daily[col] = daily[col] / 1000.0  # W·h → kWh
    
    # Self-sufficiency ratio
    daily["Self_Sufficiency_%"] = (daily["Self_Consumed_W"] / daily["Load_W"] * 100).fillna(0)
    
    return df, daily


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="☀️ PV System Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ----------------------------
    # Sidebar Configuration
    # ----------------------------
    st.sidebar.title("🔧 System Configuration")
    st.sidebar.markdown("---")
    
    # Location
    st.sidebar.subheader("📍 Location")
    city = st.sidebar.selectbox(
        "Select City",
        list(SWISS_CITIES.keys()),
        help="Select a Swiss city for weather data"
    )
    latitude, longitude, timezone = SWISS_CITIES[city]
    
    st.sidebar.caption(f"📐 Coordinates: {latitude:.4f}°N, {longitude:.4f}°E")
    
    st.sidebar.markdown("---")
    
    # PV System Parameters
    st.sidebar.subheader("☀️ PV System")
    
    cfg = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "slope": st.sidebar.slider(
            "Panel Tilt Angle (°)",
            min_value=0,
            max_value=90,
            value=int(abs(latitude)),  # Default to latitude
            help=f"Optimal tilt ≈ {int(abs(latitude))}° for {city}"
        ),
        "num_cells": st.sidebar.number_input(
            "Number of Solar Panels",
            min_value=1,
            max_value=500,
            value=20,
            help="Typical residential: 15-30 panels"
        ),
        "cell_area": st.sidebar.number_input(
            "Panel Area (m²)",
            min_value=0.1,
            max_value=3.0,
            value=1.7,
            step=0.1,
            help="Standard panel: ~1.6-1.7 m²"
        ),
        "efficiency": st.sidebar.number_input(
            "PV Efficiency",
            min_value=0.05,
            max_value=0.30,
            value=0.20,
            step=0.01,
            format="%.2f",
            help="Monocrystalline: 0.18-0.22, Polycrystalline: 0.15-0.17"
        ),
    }
    
    # Display total capacity
    peak_power_kw = cfg["num_cells"] * cfg["cell_area"] * cfg["efficiency"]
    st.sidebar.info(f"🔆 Peak Power: **{peak_power_kw:.2f} kWp**")
    
    st.sidebar.markdown("---")
    
    # Battery
    st.sidebar.subheader("🔋 Battery Storage")
    cfg["battery_capacity"] = st.sidebar.number_input(
        "Battery Capacity (kWh)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Typical residential: 5-15 kWh"
    )
    
    st.sidebar.caption(f"⚡ Charge/Discharge Efficiency: {CHARGE_EFFICIENCY*100:.0f}%")
    
    st.sidebar.markdown("---")
    
    # Load
    st.sidebar.subheader("🏠 Household Load")
    cfg["avg_load"] = st.sidebar.number_input(
        "Average Load (W)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100,
        help="Typical household: 1500-3000 W average"
    )
    
    daily_consumption_kwh = cfg["avg_load"] * 24 / 1000
    st.sidebar.info(f"📊 Daily Consumption: **{daily_consumption_kwh:.1f} kWh**")
    
    st.sidebar.markdown("---")
    
    # Date Range
    st.sidebar.subheader("📅 Simulation Period")
    cfg["start_date"] = st.sidebar.date_input(
        "Start Date",
        value=datetime(2024, 6, 1),
        min_value=MIN_DATE,
        max_value=MAX_DATE,
        help=f"Available: {MIN_DATE.date()} to {MAX_DATE.date()}"
    )
    cfg["end_date"] = st.sidebar.date_input(
        "End Date",
        value=datetime(2024, 6, 7),
        min_value=MIN_DATE,
        max_value=MAX_DATE
    )
    
    st.sidebar.markdown("---")
    
    # Run button
    run = st.sidebar.button("🚀 Run Simulation", type="primary", width='stretch')
    
    # ----------------------------
    # Main Display
    # ----------------------------
    st.title("☀️ Photovoltaic System Simulator")
    st.markdown("### Real-time PV performance analysis using NASA POWER satellite data")
    
    # Info expander
    with st.expander("ℹ️ About this simulator", expanded=False):
        st.markdown("""
        This tool simulates a residential solar photovoltaic (PV) system with battery storage.
        
        **Key Features:**
        - 📡 Real hourly irradiance data from NASA POWER API
        - 🧭 Multi-orientation panel modeling (N/S/E/W)
        - 🔋 Battery storage with efficiency losses
        - 📊 Self-consumption and grid interaction analysis
        
        **Glossary:**
        - **GHI**: Global Horizontal Irradiance - solar power hitting a horizontal surface (W/m²)
        - **SOC**: State of Charge - current battery energy level (kWh)
        - **Self-sufficiency**: Percentage of load met by solar+battery (%)
        - **kWp**: Kilowatt-peak - maximum PV system output under ideal conditions
        """)
    
    # ----------------------------
    # Run Simulation
    # ----------------------------
    if run:
        try:
            # Validate inputs
            validate_inputs(cfg)
            
            # Fetch data
            df = fetch_nasa_data(
                cfg["latitude"],
                cfg["longitude"],
                cfg["start_date"],
                cfg["end_date"]
            )
            
            if df is None or df.empty:
                st.stop()
            
            # Run simulation
            with st.spinner("⚙️ Running PV simulation..."):
                df, daily = simulate_pv_system(cfg, df)
            
            st.success("✅ Simulation complete!")
            
            # ----------------------------
            # Summary Metrics
            # ----------------------------
            st.markdown("### 📈 Summary Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Solar Irradiance",
                    f"{df['GHI'].mean():.1f} W/m²",
                    help="Average Global Horizontal Irradiance"
                )
            
            with col2:
                st.metric(
                    "Total Energy Generated",
                    f"{df['Power_Total'].sum() / 1000:.1f} kWh",
                    help="Total PV energy production over period"
                )
            
            with col3:
                st.metric(
                    "Total Load Consumed",
                    f"{df['Load_W'].sum() / 1000:.1f} kWh",
                    help="Total household energy consumption"
                )
            
            with col4:
                avg_self_sufficiency = daily["Self_Sufficiency_%"].mean()
                st.metric(
                    "Avg Self-Sufficiency",
                    f"{avg_self_sufficiency:.1f}%",
                    help="Average percentage of load met by solar+battery"
                )
            
            # ----------------------------
            # Detailed Metrics
            # ----------------------------
            st.markdown("### 🔍 Energy Balance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self_consumed = df["Self_Consumed_W"].sum() / 1000
                st.metric("Self-Consumed", f"{self_consumed:.1f} kWh")
            
            with col2:
                grid_import = df["Grid_Import_W"].sum() / 1000
                st.metric("Grid Import", f"{grid_import:.1f} kWh")
            
            with col3:
                grid_export = df["Grid_Export_W"].sum() / 1000
                st.metric("Grid Export", f"{grid_export:.1f} kWh")
            
            # ----------------------------
            # Visualizations
            # ----------------------------
            st.markdown("---")
            st.markdown("### 📊 Power Analysis")
            
            # Tab organization
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "⚡ Power Flow",
                "🧭 By Orientation",
                "🔋 Battery SOC",
                "📅 Daily Energy",
                "📈 Self-Sufficiency",
                "🔌 Grid Interaction"
            ])
            
            with tab1:
                st.markdown("#### Total PV Power vs Household Load")
                fig = create_power_flow_plot(df)
                create_plot_with_download(fig, f"power_flow_{city}_{cfg['start_date']}", "Power Flow")
                st.caption("**Net Power**: Positive = excess (charging battery/export), Negative = deficit (grid import)")
            
            with tab2:
                st.markdown("#### Power Output by Panel Orientation")
                fig = create_orientation_plot(df)
                create_plot_with_download(fig, f"orientation_{city}_{cfg['start_date']}", "Orientation")
                st.caption("Shows how different panel orientations perform throughout the day")
            
            with tab3:
                st.markdown("#### Battery State of Charge (SOC)")
                fig = create_battery_plot(df, cfg["battery_capacity"])
                create_plot_with_download(fig, f"battery_soc_{city}_{cfg['start_date']}", "Battery SOC")
                st.caption(f"Battery capacity: {cfg['battery_capacity']:.1f} kWh")
                
                # Battery statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min SOC", f"{df['SOC_kWh'].min():.2f} kWh")
                with col2:
                    st.metric("Max SOC", f"{df['SOC_kWh'].max():.2f} kWh")
                with col3:
                    st.metric("Avg SOC", f"{df['SOC_kWh'].mean():.2f} kWh")
            
            with tab4:
                st.markdown("#### Daily Energy Production vs Consumption")
                fig = create_daily_energy_plot(daily)
                create_plot_with_download(fig, f"daily_energy_{city}_{cfg['start_date']}", "Daily Energy")
                st.caption("Comparison of daily PV generation and household consumption")
            
            with tab5:
                st.markdown("#### Daily Self-Sufficiency Ratio")
                fig = create_self_sufficiency_plot(daily)
                create_plot_with_download(fig, f"self_sufficiency_{city}_{cfg['start_date']}", "Self-Sufficiency")
                st.caption("Percentage of daily load met by solar + battery (higher is better)")
            
            with tab6:
                st.markdown("#### Grid Import vs Export")
                fig = create_grid_interaction_plot(daily)
                create_plot_with_download(fig, f"grid_interaction_{city}_{cfg['start_date']}", "Grid Interaction")
                st.caption("Energy exchanged with the grid: Import (red, below zero) vs Export (green, above zero)")

            
            # ----------------------------
            # Data Table
            # ----------------------------
            st.markdown("---")
            with st.expander("📋 View Daily Summary Table"):
                st.dataframe(
                    daily.style.format({
                        "Power_Total": "{:.2f} kWh",
                        "Load_W": "{:.2f} kWh",
                        "Self_Consumed_W": "{:.2f} kWh",
                        "Grid_Import_W": "{:.2f} kWh",
                        "Grid_Export_W": "{:.2f} kWh",
                        "Self_Sufficiency_%": "{:.1f}%",
                        "GHI": "{:.1f} W/m²",
                        "Temp": "{:.1f}°C",
                    }),
                    width='stretch'
                )
            
            # ----------------------------
            # Export Options
            # ----------------------------
            st.markdown("---")
            st.markdown("### 💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly data
                csv_hourly = io.StringIO()
                df.to_csv(csv_hourly, index=True)
                st.download_button(
                    label="📥 Download Hourly Data (CSV)",
                    data=csv_hourly.getvalue(),
                    file_name=f"pv_simulation_hourly_{city}_{cfg['start_date']}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
            with col2:
                # Daily summary
                csv_daily = io.StringIO()
                daily.to_csv(csv_daily, index=True)
                st.download_button(
                    label="📥 Download Daily Summary (CSV)",
                    data=csv_daily.getvalue(),
                    file_name=f"pv_simulation_daily_{city}_{cfg['start_date']}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
        except ValidationError as e:
            st.error(f"❌ Validation Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.exception(e)
    
    else:
        # Welcome message
        st.info("👈 Configure your PV system in the sidebar and click **Run Simulation** to begin")
        
        # Quick tips
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        1. **Location**: Select your Swiss city for accurate weather data
        2. **Tilt Angle**: Default optimal tilt equals latitude (~30-47° for Switzerland)
        3. **Panel Area**: Standard residential panels are ~1.6-1.7 m²
        4. **Efficiency**: Modern panels: 18-22% (monocrystalline), 15-17% (polycrystalline)
        5. **Battery**: Typical residential storage: 5-15 kWh
        6. **Load**: Average Swiss household: 1500-3000 W
        """)


if __name__ == "__main__":
    main()
