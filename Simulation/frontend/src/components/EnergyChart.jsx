import React from "react";
import {Line} from "react-chartjs-2";

export default function EnergyChart({dates,data}) {
  const chartData={
    labels:dates,
    datasets:[{label:"Daily PV (kWh)",data,tension:0.3,fill:true,borderColor:"#2196f3"}]
  };
  const opts={responsive:true,plugins:{legend:{position:'top'}}};
  return <Line data={chartData} options={opts}/>;
}
