# ☀️ Photovoltaic System Simulator

A professional-grade Streamlit application for simulating residential solar PV systems with battery storage using real NASA satellite data.

## Features

✨ **Real Data** - Hourly irradiance from NASA POWER API (1984-present)  
🧭 **Multi-Orientation** - Simulate panels facing N/S/E/W  
🔋 **Battery Storage** - Model charge/discharge with efficiency losses  
📊 **Rich Analytics** - Self-sufficiency, grid interaction, energy balance  
💾 **Export Everything** - Download plots as PNG and data as CSV  
🎨 **Interactive Plots** - Professional Plotly visualizations  

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run pv_simulator_refined.py
```

### Requirements
- Python 3.8+
- streamlit
- pandas
- numpy
- requests
- plotly
- kaleido (for PNG export)

## Quick Start

1. **Select Location** - Choose a Swiss city from the sidebar
2. **Configure System** - Set panel specs, battery size, household load
3. **Choose Date Range** - Pick simulation period (max 365 days)
4. **Run Simulation** - Click the "🚀 Run Simulation" button
5. **Explore Results** - View interactive charts across 6 tabs
6. **Export Data** - Download plots as PNG or data as CSV

## Configuration Guide

### PV System
- **Panel Tilt**: Set to latitude for optimal year-round performance
- **Number of Panels**: Typical residential = 15-30 panels
- **Panel Area**: Standard = 1.6-1.7 m²
- **Efficiency**: Monocrystalline = 18-22%, Polycrystalline = 15-17%

### Battery Storage
- **Capacity**: Typical residential = 5-15 kWh
- **Efficiency**: Fixed at 95% charge/discharge (realistic)

### Household Load
- **Average Load**: Typical Swiss household = 1500-3000 W
- **Daily Consumption**: Automatically calculated

## Understanding the Charts

### 1. Power Flow
Shows PV generation, household load, and net power (excess/deficit)

### 2. Orientation Analysis
Compares performance of panels facing different directions

### 3. Battery SOC
Tracks battery state of charge throughout the simulation

### 4. Daily Energy
Bar chart comparing daily generation vs consumption

### 5. Self-Sufficiency
Percentage of load met by solar + battery (goal: >75%)

### 6. Grid Interaction
Shows energy imported from and exported to the grid

## Export Options

**PNG Plots**: Click "💾 Save as PNG" under any chart  
**CSV Data**: Download hourly or daily summary data  

All exports include location and date in filename.

## Data Source

This simulator uses the [NASA POWER API](https://power.larc.nasa.gov/) which provides:
- **GHI**: Global Horizontal Irradiance from satellite observations
- **Temperature**: 2m air temperature
- **Coverage**: Global, 1984-present
- **Resolution**: Hourly

## Technical Notes

- Tilt factor uses latitude-optimized model
- Orientation factors based on Swiss latitude (46-48°N)
- Battery model includes realistic 95% round-trip efficiency
- Load profile based on typical residential patterns
- Results cached for 1 hour to minimize API calls

## Limitations

- Simplified orientation model (doesn't include full solar path)
- Fixed charge/discharge efficiency (real batteries vary)
- No seasonal tilt adjustment
- No shading or soiling losses
- No inverter losses modeled

## License

MIT License - Feel free to use and modify

## Credits

Built with Streamlit, Plotly, and NASA POWER data
