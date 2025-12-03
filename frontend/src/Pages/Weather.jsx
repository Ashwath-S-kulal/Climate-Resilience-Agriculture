import React, { useEffect, useState, useRef } from "react";
import {
  Search, MapPin, Thermometer, Droplet, Wind, Sun, CloudSun, Cloud, CloudFog, CloudDrizzle,
  CloudRain, CloudSnow, CloudLightning, CloudOff, Calendar, TrendingUp, TrendingDown, BarChart3
} from 'lucide-react';
import Header from "../Components/Header";
import { FiSearch } from "react-icons/fi";
import ChatbotIcon from "../Components/ChatbotIcon";

// --- Helper Functions (unchanged) ---
const getWeatherIcon = (code, className = "w-12 h-12 text-green-700") => {
  let color = "text-yellow-500";
  if (code >= 61 && code < 80) color = "text-blue-500";
  else if (code >= 71 && code < 90) color = "text-gray-400";
  else if (code >= 95) color = "text-red-500";
  const finalClassName = `${className} ${color}`;
  switch (code) {
    case 0: return <Sun className={finalClassName} />;
    case 1:
    case 2: return <CloudSun className={finalClassName} />;
    case 3: return <Cloud className={finalClassName} />;
    case 45:
    case 48: return <CloudFog className={finalClassName} />;
    case 51:
    case 53:
    case 55: return <CloudDrizzle className={finalClassName} />;
    case 61:
    case 63:
    case 65: return <CloudRain className={finalClassName} />;
    case 71:
    case 73:
    case 75: return <CloudSnow className={finalClassName} />;
    case 80:
    case 81:
    case 82: return <CloudRain className={finalClassName} />;
    case 95: return <CloudLightning className={finalClassName} />;
    default: return <CloudOff className={finalClassName} />;
  }
};

const weatherCodeText = (code) => {
  const map = {
    0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime Fog", 51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Heavy Drizzle",
    61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain", 71: "Slight Snow", 73: "Moderate Snow",
    75: "Heavy Snow", 80: "Slight Showers", 82: "Heavy Showers", 95: "Thunderstorm",
  };
  return map[code] || "Unspecified Weather";
};

export default function Weather() {
  const [city, setCity] = useState("");
  const [coords, setCoords] = useState(null);
  const [locationName, setLocationName] = useState("Getting your location...");
  const [today, setToday] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const searchTimeout = useRef(null);

  useEffect(() => { getCurrentLocation(); }, []);
  useEffect(() => { if (coords) { getWeather(coords.lat, coords.lon); getHistory(coords.lat, coords.lon); } }, [coords]);

  // --- Current Location ---
  const getCurrentLocation = () => {
    setLoading(true); setError("");
    if (!navigator.geolocation) {
      setError("Geolocation not supported");
      setLoading(false); setLocationName("Please search for a city.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        const lat = coords.latitude, lon = coords.longitude;
        setCoords({ lat, lon });
        reverseGeocode(lat, lon);
        setLoading(false);
      },
      () => { setError("Location access blocked — search manually"); setLoading(false); setLocationName("Location access denied."); }
    );
  };

  // --- Reverse Geocode ---
  const reverseGeocode = async (lat, lon) => {
    try {
      const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}`;
      const res = await fetch(url);
      const data = await res.json();
      if (data && data.address) {
        const { village, town, city, state, country } = data.address;
        const name = city || town || village || "";
        setLocationName(`${name || 'Near Coordinates'}, ${state || 'State'}, ${country || 'Country'}`);
      }
    } catch { setLocationName(`Location at ${lat.toFixed(2)}, ${lon.toFixed(2)}`); }
  };

  // --- City Search ---
  const searchCity = async (selectedCity) => {
    const query = selectedCity || city;
    if (!query) return;
    setLoading(true);
    setSearching(true);
    setError("");
    setSuggestions([]);
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data || !data.length) {
        setError(`City "${query}" not found!`);
        setLoading(false); setSearching(false);
        return;
      }
      const lat = data[0].lat, lon = data[0].lon;
      setCoords({ lat, lon });
      reverseGeocode(lat, lon);
    } catch (e) { setError("Failed to search city. Check network."); console.error(e); }
    finally { setLoading(false); setSearching(false); }
  };

  // --- Fetch Suggestions ---
  const fetchSuggestions = (value) => {
    clearTimeout(searchTimeout.current);
    setCity(value);
    if (!value) { setSuggestions([]); return; }
    searchTimeout.current = setTimeout(async () => {
      try {
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(value)}&limit=5`;
        const res = await fetch(url);
        const data = await res.json();
        setSuggestions(data.map(item => ({ name: item.display_name, lat: item.lat, lon: item.lon })));
      } catch { setSuggestions([]); }
    }, 300); // debounce 300ms
  };

  // --- Weather & History (unchanged) ---
  const getWeather = async (lat, lon) => {
    try {
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
        `&current=temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m` +
        `&daily=temperature_2m_max,temperature_2m_min,weather_code` +
        `&timezone=auto&temperature_unit=celsius&wind_speed_unit=ms`;
      const res = await fetch(url);
      const data = await res.json();
      setToday({ temp: data.current.temperature_2m, humidity: data.current.relative_humidity_2m, wind: data.current.wind_speed_10m, weather_code: data.current.weather_code });
      const d = data.daily.time.map((t, i) => ({ date: t, temp_max: data.daily.temperature_2m_max[i], temp_min: data.daily.temperature_2m_min[i], weather_code: data.daily.weather_code[i] }));
      setForecast(d.slice(1, 8));
    } catch (e) { setError("Failed to fetch current weather."); console.error(e); }
  };

  const getHistory = async (lat, lon) => {
    try {
      const end = new Date(), start = new Date();
      start.setDate(start.getDate() - 30);
      const format = (d) => d.toISOString().split("T")[0];
      const url = `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}` +
        `&start_date=${format(start)}&end_date=${format(end)}` +
        `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max` +
        `&timezone=auto&temperature_unit=celsius&wind_speed_unit=ms`;
      const res = await fetch(url);
      const data = await res.json();
      const arr = data.daily.time.map((t, i) => ({ date: t, temp_max: data.daily.temperature_2m_max[i], temp_min: data.daily.temperature_2m_min[i], rain: data.daily.precipitation_sum[i], wind: data.daily.wind_speed_10m_max[i], weather_code: data.daily.weather_code[i] }));
      setHistory(arr.reverse());
    } catch (e) { console.log("Failed to fetch history.", e); }
  };

  const todayDate = today ? new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }) : null;

  return (
    <div className="min-h-screen bg-[#95d5b2] text-gray-800 font-sans">
      <div className="bg-gradient-to-t from-green-800 to-green-600 text-white shadow-lg pb-1 md:pb-6 pt-16 md:pt-24 sm:pt-24 md:rounded-b-3xl">
        <Header className="bg-transparent" />
        <div className="max-w-7xl mx-auto px-4 sm:px-8">
          <header className="mb-8">
            <h1 className="text-4xl font-extrabold tracking-wide mb-2 justify-self-center">
              Live Weather Forecast
            </h1>
            <p className="text-green-200 text-xs md:text-base justify-self-center ">
              Stay informed with reliable weather updates designed to support smarter field management and farm planning.
            </p>

            <div className="relative mt-6 w-full">
              <div className="flex gap-2 bg-white rounded-full shadow-xl px-4 py-2 items-center w-full">
                <FiSearch className="text-green-500 text-lg pointer-events-none" />
                <input
                  type="text"
                  placeholder="Search Farm, Town or Region..."
                  className="flex-grow px-2 py-2 bg-transparent text-gray-800 placeholder-gray-500 focus:ring-0 outline-none text-base"
                  value={city}
                  onChange={(e) => fetchSuggestions(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && searchCity()}
                />
                <button
                  onClick={() => searchCity()}
                  className="px-5 py-2 bg-lime-400 hover:bg-lime-300 rounded-full transition-all text-green-900 font-semibold"
                >
                  <FiSearch className="w-5 h-5" />
                </button>
              </div>

              {suggestions.length > 0 && (
                <ul className="absolute z-50 w-full mt-1 bg-white border border-green-200 rounded-md shadow-lg max-h-60 overflow-auto">
                  {suggestions.map((s, i) => (
                    <li
                      key={i}
                      onClick={() => searchCity(s.name)}
                      className="px-4 py-2 hover:bg-green-100 cursor-pointer text-gray-800"
                    >
                      {s.name}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </header>
        </div>
      </div>

      <div className="w-full max-w-7xl mx-auto px-4 sm:px-8 pt-0 mt-9 py-10 bg-[#95d5b2]">
        {loading && (
          <div className="flex flex-col justify-center items-center py-12">
            <div className="w-12 h-12 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-md font-medium text-green-600 mt-3">
              {searching ? "Searching for city..." : "Checking the fields..."}
            </p>
          </div>
        )}
        {!loading && (
          <div>
            {error && (
              <p className="mb-8 p-4 bg-red-100 text-red-700 rounded-lg border border-red-400 font-medium text-sm">
                Error: {error}
              </p>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 w-full ">

              {today && (
                <div className="lg:col-span-1 space-y-8 w-full">
                  <div className="p-6 bg-white rounded-xl shadow-xl border border-green-200 w-full">
                    <h2 className="text-2xl font-bold mb-6 flex items-center text-green-700 border-b border-green-100 pb-4">
                      <Calendar className="w-6 h-6 mr-3 text-lime-600" />
                      Current Weather
                    </h2>

                    <h2 className="text-2xl font-bold mb-1 flex items-center text-green-800">
                      <MapPin className="w-5 h-5 mr-2 text-green-600" />
                      {locationName.split(",")[0].trim()}
                    </h2>

                    <p className="text-sm mb-4 text-gray-500 flex items-center">
                      <Calendar className="w-4 h-4 mr-1 text-gray-400" /> {todayDate}
                    </p>

                    <div className="flex flex-col items-center mb-6 border-b border-green-100 pb-6">
                      <div className="text-green-700">
                        {getWeatherIcon(today.weather_code, "w-28 h-28")}
                      </div>

                      <p className="text-7xl font-extrabold text-green-900 mt-4">
                        {Math.round(today.temp)}
                        <span className="text-4xl text-lime-600">°C</span>
                      </p>

                      <p className="text-lg font-semibold mt-1 text-gray-700">
                        {weatherCodeText(today.weather_code)}
                      </p>
                    </div>

                    <div className="grid grid-cols-3 gap-5">
                      <div className="p-3 bg-green-50 rounded-lg text-center border border-green-200">
                        <Thermometer className="w-7 h-7 mx-auto text-red-500" />
                        <p className="text-xs mt-1 text-gray-500">Temperature</p>
                        <p className="md:text-xl font-bold text-gray-800">{today.temp.toFixed(1)}°C</p>
                      </div>

                      <div className="p-3 bg-green-50 rounded-lg text-center border border-green-200">
                        <Droplet className="w-7 h-7 mx-auto text-blue-500" />
                        <p className="text-xs mt-1 text-gray-500">Humidity</p>
                        <p className="md:text-xl font-bold text-gray-800">{today.humidity}%</p>
                      </div>

                      <div className="p-3 bg-green-50 rounded-lg text-center border border-green-200">
                        <Wind className="w-7 h-7 mx-auto text-lime-600" />
                        <p className="text-xs mt-1 text-gray-500">WindSpeed</p>
                        <p className="md:text-xl font-bold text-gray-800">{today.wind.toFixed(1)} m/s</p>
                      </div>
                    </div>

                    <div className="bg-green-400/10 mt-3 px-10 p-2 rounded-md text-gray-600 italic text-sm">
                      <p>Local weather may differ slightly.</p>
                    </div>

                  </div>
                </div>
              )}

              <div className="lg:col-span-2 w-full">
                {forecast.length > 0 && (
                  <div className="p-3 md:p-6 bg-white rounded-xl shadow-xl border border-green-200 w-full">
                    <h2 className="text-lg md:text-2xl font-bold mb-6 flex items-center text-green-700 border-b border-green-100 pb-4">
                      <Calendar className="w-6 h-6 mr-3 text-lime-600" />
                      7-Day Weather Forecast
                    </h2>

                    <div className="space-y-3">
                      {forecast.map((day, i) => {
                        const dayName = new Date(day.date).toLocaleDateString("en-US", { weekday: "long" });
                        const date = new Date(day.date).toLocaleDateString("en-US", { month: "short", day: "numeric" });
                        return (
                          <div
                            key={i}
                            className="flex items-center p-4 rounded-lg bg-green-50 hover:bg-green-100 transition border border-green-100 hover:border-lime-300"
                          >
                            <div className="w-20 md:w-36">
                              <p className="font-bold text-gray-800 text-xs md:text-base">{dayName}</p>
                              <p className="text-xs text-gray-500">{date}</p>
                            </div>

                            <div className="flex justify-start items-center md:flex-grow  md:mx-4">
                              {getWeatherIcon(day.weather_code, "w-4 md:w-7 h-4 md:h-7 mr-2 md:mr-3")}
                              <p className="text-xs md:text-sm text-gray-700 font-medium">
                                {weatherCodeText(day.weather_code)}
                              </p>
                            </div>

                            <div className="flex space-x-3 md:space-x-8 ml-auto text-right font-bold text-xs md:text-lg">
                              <p className="text-red-500 flex items-center">
                                <TrendingUp className="w-4 h-4 mr-1 text-red-400" /> {Math.round(day.temp_max)}°
                              </p>
                              <p className="text-blue-500 flex items-center">
                                <TrendingDown className="w-4 h-4 mr-1 text-blue-400" /> {Math.round(day.temp_min)}°
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {history.length > 0 && (
              <div className="p-2 md:p-6 bg-white rounded-xl shadow-xl border border-green-200 mt-10 w-full">
                <h2 className="text-xl md:text-2xl font-bold mb-6 flex items-center text-green-700 border-b border-green-100 pb-4">
                  <BarChart3 className="w-6 h-6 mr-3 text-lime-600" />
                  30-Day Historical Data
                </h2>

                <div className="overflow-x-auto rounded-lg border border-green-200">
                  <table className="min-w-full divide-y divide-green-200 text-sm">
                    <thead className="bg-green-50">
                      <tr className="text-left text-xs md:text-sm font-semibold text-green-600 uppercase tracking-wider">
                        <th className="px-4 py-3">Date</th>
                        <th className="px-4 py-3 text-center">Condition</th>
                        <th className="px-4 py-3 text-right">Max(°C)</th>
                        <th className="px-4 py-3 text-right">Min(°C)</th>
                        <th className="px-4 py-3 text-right">Rain(mm)</th>
                        <th className="px-4 py-3 text-right">Wind(m/s)</th>
                      </tr>
                    </thead>

                    <tbody className="divide-y divide-green-100">
                      {history.map((day, i) => (
                        <tr key={i} className="bg-white hover:bg-green-50 text-xs md:text-sm">
                          <td className="px-4 py-3 text-gray-800 font-medium">{day.date}</td>
                          <td className="px-4 py-3 text-center text-gray-700">
                            <span className="flex items-center justify-center">
                              {getWeatherIcon(day.weather_code, "w-5 h-5 mr-2")}
                              {weatherCodeText(day.weather_code)}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right text-red-500 font-bold">{day.temp_max}</td>
                          <td className="px-4 py-3 text-right text-blue-500 font-bold">{day.temp_min}</td>
                          <td className="px-4 py-3 text-right text-blue-600 font-bold">{day.rain}</td>
                          <td className="px-4 py-3 text-right text-gray-700 font-medium">{day.wind}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      <ChatbotIcon />
    </div>
  );
}
