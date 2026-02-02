import numpy as np
import pandas as pd

class Battery:
    def __init__(self, capacity_kwh, max_charge_kw, max_discharge_kw,
                 soc_init=0.5, round_trip_eff=0.9,
                 soc_min=0.1, soc_max=0.95):
        self.capacity = capacity_kwh
        self.max_charge = max_charge_kw
        self.max_discharge = max_discharge_kw
        self.soc = soc_init * capacity_kwh
        self.rt_eff = round_trip_eff
        self.soc_min = soc_min * capacity_kwh
        self.soc_max = soc_max * capacity_kwh

    def step(self, pv_kwh, load_kwh=0):
        res = dict(pv_used_direct=0, charged=0, discharged=0,
                   curtailed=0, grid_draw=0, soc_end=0)
        pv = pv_kwh
        supply = min(pv, load_kwh)
        pv -= supply
        load = load_kwh - supply
        if load>0:
            dis = min(self.max_discharge, self.soc - self.soc_min, load)
            self.soc -= dis
            res['discharged']=dis
            load -= dis
            if load>0: res['grid_draw']=load
        chg = min(self.max_charge, self.soc_max - self.soc,
                  pv*self.rt_eff)
        pv_used = chg/self.rt_eff if self.rt_eff>0 else 0
        if pv_used>pv: pv_used=pv; chg=pv*self.rt_eff
        self.soc += chg
        pv -= pv_used
        res.update(dict(charged=chg, curtailed=pv,
                        pv_used_direct=pv_kwh-(chg+pv),
                        soc_end=self.soc))
        return res

def simulate_battery_series(pv_series, battery, load_series=None):
    if load_series is None:
        load_series=[0]*len(pv_series)
    out=[]
    for pv,load in zip(pv_series,load_series):
        rec=battery.step(pv,load)
        rec.update(pv_kwh=pv, load_kwh=load, soc_kwh=battery.soc)
        out.append(rec)
    return pd.DataFrame(out)
