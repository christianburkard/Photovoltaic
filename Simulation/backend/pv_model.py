import numpy as np
import pandas as pd

# ---- Simple daily model ----
def tilt_factor(tilt_deg, lat_deg, day_of_year):
    tilt = np.radians(tilt_deg)
    lat = np.radians(lat_deg)
    decl = np.radians(23.45)*np.sin(2*np.pi*(284+day_of_year)/365)
    solar_alt = np.arcsin(np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl))
    proj = np.clip(np.cos(np.abs(tilt - solar_alt)), 0, 1)
    diffuse = 0.5*(1+np.cos(tilt))
    ground = 0.2*(1-np.cos(tilt))/2
    return float((proj+diffuse+ground)/1.5)

def estimate_daily_pv_energy(irr_wm2, day_of_year, tilt, lat,
                             num_cells, cell_area, eff):
    area = num_cells * cell_area
    wh_m2 = irr_wm2 * 24
    tf = tilt_factor(tilt, lat, day_of_year)
    kwh = wh_m2 * tf * area * eff / 1000
    return kwh

# ---- pvlib hourly model ----
try:
    import pvlib
    PVLIB_AVAILABLE = True
except Exception:
    PVLIB_AVAILABLE = False

def estimate_daily_pv_energy_hourly(lat, lon, tilt, num_cells,
                                    cell_area, eff, start, end):
    """Hourly pvlib-based model using NASA POWER hourly irradiance"""
    if not PVLIB_AVAILABLE:
        raise RuntimeError("pvlib not installed")

    import requests
    NASA_HOURLY = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "start": start.replace('-',''),
        "end": end.replace('-',''),
        "latitude": lat,
        "longitude": lon,
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "format": "JSON",
        "time-standard": "UTC"
    }
    r = requests.get(NASA_HOURLY, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js['properties']['parameter'])
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H', utc=True)
    ghi = df['ALLSKY_SFC_SW_DWN'].astype(float)

    loc = pvlib.location.Location(lat, lon)
    solpos = loc.get_solarposition(ghi.index)
    dni = pvlib.irradiance.dirint(ghi, solpos['zenith'], ghi.index)
    dhi = ghi - dni*np.cos(np.radians(solpos['zenith']))
    dhi[dhi<0] = 0

    poa = pvlib.irradiance.get_total_irradiance(
        tilt, 180, dni, ghi, dhi,
        solpos['zenith'], solpos['azimuth'], model='haydavies'
    )
    poa_series = pd.Series(poa['poa_global'], index=ghi.index)
    daily = poa_series.groupby(poa_series.index.date).sum()  # Wh/m2/day
    area = num_cells * cell_area
    kwh = daily.values * area * eff / 1000
    dates = [d.strftime("%Y-%m-%d") for d in daily.index]
    return dates, kwh.tolist()
