import React,{useState} from "react";
import PVForm from "./components/PVForm";
import EnergyChart from "./components/EnergyChart";
import {postEstimate,postPlot} from "./api";

export default function App(){
  const [dates,setDates]=useState([]);
  const [pv,setPv]=useState([]);
  const [plot,setPlot]=useState(null);

  const handleSubmit=async form=>{
    const payload={...form,battery:{capacity_kwh:form.battery_capacity_kwh}};
    const r=await postEstimate(payload);
    const j=r.data;
    setDates(j.dates); setPv(j.daily_pv_kwh);
    const img=await postPlot(payload);
    const url=URL.createObjectURL(img.data);
    setPlot(url);
  };

  return(
  <div style={{maxWidth:900,margin:"0 auto"}}>
    <h1>Solar PV & Battery Estimator</h1>
    <PVForm onSubmit={handleSubmit}/>
    {dates.length>0 && <EnergyChart dates={dates} data={pv}/>}
    {plot && <img src={plot} alt="plot" style={{width:"100%",marginTop:20}}/>}
  </div>);
}
