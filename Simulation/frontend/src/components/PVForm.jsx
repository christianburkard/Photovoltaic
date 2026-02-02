import React,{useState,useEffect} from "react";

export default function PVForm({onSubmit}) {
  const [cities,setCities]=useState([]);
  const [form,setForm]=useState({
    city:"Zürich",latitude:47.37,longitude:8.54,tilt:30,
    num_cells:20,cell_area:0.165,module_efficiency:0.18,
    start_date:"2025-01-01",end_date:"2025-01-31",battery_capacity_kwh:10
  });

  useEffect(()=>{fetch('/data/swiss_cities.json')
    .then(r=>r.json()).then(setCities).catch(console.error)},[]);

  const setField=(k,v)=>setForm(s=>({...s,[k]:v}));

  const onCityChange=e=>{
    const c=cities.find(x=>x.name===e.target.value);
    setField("city",e.target.value);
    if(c){setField("latitude",c.lat);setField("longitude",c.lon);}
  };

  return(
  <form onSubmit={e=>{e.preventDefault();onSubmit(form);}}>
    <label>City</label>
    <select value={form.city} onChange={onCityChange}>
      {cities.map(c=><option key={c.name}>{c.name}</option>)}
    </select>
    <label>Latitude</label><input type="number"
      value={form.latitude} onChange={e=>setField("latitude",+e.target.value)}/>
    <label>Longitude</label><input type="number"
      value={form.longitude} onChange={e=>setField("longitude",+e.target.value)}/>
    <label>Tilt</label><input type="number"
      value={form.tilt} onChange={e=>setField("tilt",+e.target.value)}/>
    <label>Cells</label><input type="number"
      value={form.num_cells} onChange={e=>setField("num_cells",+e.target.value)}/>
    <label>Area</label><input type="number"
      value={form.cell_area} onChange={e=>setField("cell_area",+e.target.value)}/>
    <label>Efficiency</label><input type="number"
      value={form.module_efficiency} step="0.01"
      onChange={e=>setField("module_efficiency",+e.target.value)}/>
    <label>Start</label><input type="date"
      value={form.start_date} onChange={e=>setField("start_date",e.target.value)}/>
    <label>End</label><input type="date"
      value={form.end_date} onChange={e=>setField("end_date",e.target.value)}/>
    <label>Battery (kWh)</label><input type="number"
      value={form.battery_capacity_kwh}
      onChange={e=>setField("battery_capacity_kwh",+e.target.value)}/>
    <button type="submit">Estimate</button>
  </form>);
}
