from flask import Flask, request, jsonify, send_file
import requests, pandas as pd, numpy as np, io, matplotlib.pyplot as plt
from datetime import datetime
from pv_model import estimate_daily_pv_energy, estimate_daily_pv_energy_hourly
from battery_sim import Battery, simulate_battery_series

app = Flask(__name__)

# --- Built-in NASA POWER data fetcher ---
def fetch_nasa_power(lat, lon, start_date, end_date,
                     parameters=None, tz='UTC'):
    """Fetch daily NASA POWER data for given location and date range."""
    NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"
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

# --- PV + Battery logic ---
def compute_series(payload, use_hourly=True):
    lat=float(payload["latitude"])
    lon=float(payload["longitude"])
    tilt=float(payload.get("tilt",30))
    n=int(payload.get("num_cells",20))
    area=float(payload.get("cell_area",0.165))
    eff=float(payload.get("module_efficiency",0.18))
    start=payload["start_date"]
    end=payload["end_date"]

    # Try hourly pvlib model first
    if use_hourly:
        try:
            dates,prod=estimate_daily_pv_energy_hourly(lat,lon,tilt,n,area,eff,start,end)
        except Exception as e:
            print("Hourly model failed:",e)
            use_hourly=False

    # Fallback to daily NASA model
    if not use_hourly:
        df=fetch_nasa_power(lat,lon,start,end)
        irr=df["ALLSKY_SFC_SW_DWN"].reindex(pd.date_range(start,end)).fillna(method='ffill')
        dates=[]; prod=[]
        for d,v in zip(irr.index,irr):
            doy=d.timetuple().tm_yday
            prod.append(estimate_daily_pv_energy(v,doy,tilt,lat,n,area,eff))
            dates.append(d.strftime("%Y-%m-%d"))

    # Battery simulation
    batcfg=payload.get("battery",{})
    bat=Battery(float(batcfg.get("capacity_kwh",10)),
                float(batcfg.get("max_charge_kw",5)),
                float(batcfg.get("max_discharge_kw",5)),
                float(batcfg.get("soc_init",0.5)))
    sim=simulate_battery_series(prod,bat)
    return dates,prod,sim

# --- API endpoints ---
@app.route("/api/estimate",methods=["POST"])
def estimate_json():
    p=request.json
    dates,prod,sim=compute_series(p,True)
    return jsonify({
        "dates":dates,
        "daily_pv_kwh":prod,
        "battery":sim.to_dict(orient="list"),
        "average_daily_kwh":float(np.mean(prod))
    })

@app.route("/api/plot",methods=["POST"])
def plot_png():
    p=request.json
    dates,prod,sim=compute_series(p,True)
    fig,ax=plt.subplots(figsize=(10,5))
    x=pd.to_datetime(dates)
    ax.plot(x,prod,label="PV (kWh)")
    ax2=ax.twinx()
    ax2.plot(x,sim["soc_kwh"],"--",label="SOC (kWh)")
    ax.set_title(f"PV & Battery - {p.get('city','')}")
    ax.set_ylabel("Energy (kWh)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150); buf.seek(0)
    return send_file(buf,mimetype="image/png")

if __name__=="__main__":
    app.run(debug=True,port=5000)
