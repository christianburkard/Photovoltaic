import axios from "axios";
const API_BASE = "http://localhost:5000";

export function postPlot(payload) {
  return axios.post(`${API_BASE}/api/plot`, payload, {responseType:'blob'});
}

export function postEstimate(payload) {
  return axios.post(`${API_BASE}/api/estimate`, payload);
}
