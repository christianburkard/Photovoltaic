import requests
import pandas as pd

NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_nasa_power(lat, lon, start_date, end_date,
                     parameters=None, tz='UTC'):
    if parameters is None:
        parameters = ["ALLSKY_SFC_SW_DWN", "T2M"]
    params = {
        "start": start_date.replace('-', ''),
        "end": end_date.replace('-', ''),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": ",".join(parameters),
        "format": "JSON",
        "time-standard": tz
    }
    r = requests.get(NASA_POWER_DAILY, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()['properties']['parameter']
    df = pd.DataFrame.from_dict(data)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    return df
