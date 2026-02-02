import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import requests
from pv_model import estimate_daily_pv_energy
from battery_sim import Battery, simulate_battery_series

def fetch_nasa_power(lat, lon, start_date, end_date):
    """Fetch daily NASA POWER data for given location/date range."""
    NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "format": "JSON",
        "time-standard": "UTC"
    }
    r = requests.get(NASA_POWER_DAILY, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame.from_dict(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PV + Battery Estimator (Tk)")
        self.geometry("500x450")

        labels = ["Latitude", "Longitude", "Tilt", "Cells",
                  "Cell area (m²)", "Efficiency", "Start", "End", "Battery (kWh)"]
        defaults = ["47.37", "8.54", "30", "20",
                    "0.165", "0.18", "2025-01-01", "2025-01-31", "10"]
        self.vars = {}
        for i, (lbl, val) in enumerate(zip(labels, defaults)):
            tk.Label(self, text=lbl).grid(row=i, column=0, sticky="w", padx=5, pady=3)
            v = tk.StringVar(value=val)
            tk.Entry(self, textvariable=v).grid(row=i, column=1)
            self.vars[lbl] = v

        ttk.Button(self, text="Estimate", command=self.run).grid(row=len(labels),
                                                                 column=0, columnspan=2, pady=15)

    def run(self):
        try:
            lat = float(self.vars["Latitude"].get())
            lon = float(self.vars["Longitude"].get())
            tilt = float(self.vars["Tilt"].get())
            cells = int(self.vars["Cells"].get())
            area = float(self.vars["Cell area (m²)"].get())
            eff = float(self.vars["Efficiency"].get())
            start = self.vars["Start"].get()
            end = self.vars["End"].get()
            bat_kwh = float(self.vars["Battery (kWh)"].get())

            df = fetch_nasa_power(lat, lon, start, end)
            irr = df["ALLSKY_SFC_SW_DWN"].reindex(pd.date_range(start, end)).fillna(method="ffill")
            pv = [
                estimate_daily_pv_energy(v, d.timetuple().tm_yday,
                                         tilt, lat, cells, area, eff)
                for d, v in zip(irr.index, irr)
            ]
            bat = Battery(bat_kwh, 5, 5, 0.5)
            sim = simulate_battery_series(pv, bat)
            avg = sum(pv) / len(pv)
            messagebox.showinfo("Result", f"Average daily PV yield: {avg:.2f} kWh")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()
