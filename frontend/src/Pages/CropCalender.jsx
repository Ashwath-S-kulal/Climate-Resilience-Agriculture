import React, { useState } from "react";
import Header from "../Components/Header"
import { FaCalendarAlt, FaCheckCircle, FaMapMarkerAlt, FaSeedling, FaTractor } from "react-icons/fa"
import { NavLink } from "react-router-dom";
import { useSelector } from "react-redux";

export default function CropCalendar() {
  const [area, setArea] = useState("");
  const [coords, setCoords] = useState(null);

  const [cropQuery, setCropQuery] = useState("");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
    const { currentUser } = useSelector((state) => state.user);


  const monthName = (m) =>
    new Date(2020, m - 1, 1).toLocaleString("default", { month: "short" });

  // ✅ Fetch location name via Nominatim
  const getLocationName = async (lat, lon) => {
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`
      );

      const data = await res.json();
      if (data?.display_name) {
        setArea(data.display_name);
      } else {
        setArea("Unknown Location");
      }
    } catch {
      setArea("Unknown Location");
    }
  };

  // ✅ Browser geolocation
  const fetchCurrentLocation = () => {
    if (!navigator.geolocation)
      return alert("Geolocation not supported");

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        setCoords({ lat, lon });

        await getLocationName(lat, lon);
      },
      () => alert("Unable to fetch location")
    );
  };

  const run = async () => {
    if (!cropQuery) return alert("Please enter a crop");

    try {
      setLoading(true);

      const res = await fetch("/api/calender/generatecalender", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          areaName: area,
          cropType: cropQuery,
          location: coords,
        }),
      });

      const j = await res.json();
      setResult(j);
    } catch (err) {
      console.error(err);
      alert("Error fetching crop calendar");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-green-100 ">
      <Header />

      <div className="w-full bg-gradient-to-r from-green-600 to-green-700 text-white pt-24 pb-16 px-6 shadow-lg rounded-b-3xl">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-5xl md:text-5xl font-extrabold mb-4 tracking-tight">
            Climate-Smart <span className="text-green-100">Crop Calendar</span>
          </h1>
          <p className="text-green-50 text-lg md:text-xl max-w-2xl mx-auto">
            Enter your area and crop to get the best planting and harvest windows.
          </p>
        </div>
      </div>


      <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-8 mt-10">

        <div className="bg-white shadow-2xl rounded-3xl p-6 border-l-8 border-green-500 hover:scale-[1.02] transition-transform duration-300">
          <h2 className="text-2xl font-semibold text-green-700 mb-6 flex items-center gap-2">
            <FaCheckCircle /> Enter Details
          </h2>

          <input
            className="w-full p-3 border rounded-2xl focus:ring-2 focus:ring-green-500 outline-none shadow-sm"
            placeholder="Enter crop (e.g. maize)"
            value={cropQuery}
            onChange={(e) => setCropQuery(e.target.value)}
          />

          <div className="flex flex-col md:flex-row gap-3 mb-4 mt-8">
            <input
              className="flex-1 p-3 border rounded-2xl focus:ring-2 focus:ring-green-500 outline-none shadow-sm"
              placeholder="Enter Area Name"
              value={area}
              onChange={(e) => setArea(e.target.value)}
            />
            <button
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-2xl shadow transition flex items-center justify-center gap-2"
              onClick={fetchCurrentLocation}
            >
              <FaMapMarkerAlt /> Use My Location
            </button>
          </div>





          <button
            className="mt-6 w-full px-6 py-3 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-bold rounded-2xl shadow-lg transition flex justify-center items-center gap-2"
            onClick={run}
            disabled={loading}
          >
            {loading ? "Loading..." : <><FaCheckCircle /> Generate Calendar</>}
          </button>
        </div>

        <div className="bg-white p-6 rounded-3xl shadow-2xl border border-dashed border-gray-300 hover:scale-[1.02] transition duration-300">
          <h2 className="text-2xl font-bold mb-6 text-green-700 flex items-center gap-2 border-b border-green-200 pb-2">
            <FaCalendarAlt /> Suggested Months for {cropQuery || "—"}
          </h2>

          <div className="grid grid-cols-4 gap-3 mt-3">
            {result?.months?.map((m) => (
              <div
                key={m.month}
                className={`p-3 rounded-lg text-center transition transform hover:scale-105 shadow-sm ${m.suitable
                  ? "bg-green-200 border border-green-400"
                  : "bg-gray-200 border border-gray-300"
                  }`}
              >
                <div className="text-xs font-bold">{monthName(m.month)}</div>
                <div className="text-sm font-semibold">{Math.round(m.tmean)}°C</div>
                <div className="text-xs">{Math.round(m.rainfall_mm)} mm</div>
                <div className="text-[10px] font-semibold mt-1">
                  {m.suitable ? " Suitable" : " Not Ideal"}
                </div>
              </div>
            ))}

            {Array.from({ length: 12 - (result?.months?.length || 0) }).map((_, i) => (
              <div key={i} className="p-3 rounded-lg text-center bg-gray-100 border border-gray-200 text-gray-400">
                —
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto mt-8 flex flex-col md:flex-row gap-6">

        <div className="flex-1 bg-green-50 border-l-8 border-green-500 p-6 rounded-3xl shadow-lg hover:scale-[1.02] transition duration-300">
          <h3 className="font-semibold text-lg flex items-center gap-2 text-green-700 border-b border-green-200 pb-1">
            <FaSeedling /> Planting Window
          </h3>
          {result?.planting_windows?.length ? (
            <div className="flex flex-wrap gap-2 mt-3">
              {result.planting_windows.map((w, i) => (
                <span
                  key={i}
                  className="px-4 py-2 bg-green-600 text-white rounded-2xl text-sm shadow hover:bg-green-700 transition"
                >
                  {monthName(w.start_month)} → {monthName(w.end_month)}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-red-500 mt-2">No suitable planting window found.</p>
          )}
        </div>

        <div className="flex-1 bg-yellow-50 border-l-8 border-yellow-500 p-6 rounded-3xl shadow-lg hover:scale-[1.02] transition duration-300">
          <h3 className="font-semibold text-lg flex items-center gap-2 text-yellow-600 border-b border-yellow-200 pb-1">
            <FaTractor /> Harvest Window
          </h3>
          {result?.harvest_windows?.length ? (
            <div className="flex flex-wrap gap-2 mt-3">
              {result.harvest_windows.map((w, i) => (
                <span
                  key={i}
                  className="px-4 py-2 bg-yellow-500 text-white rounded-2xl text-sm shadow hover:bg-yellow-600 transition"
                >
                  {monthName(w.harvest_est_start_month)}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-red-500 mt-2">No predicted harvest windows.</p>
          )}
        </div>
      </div>
      <div className="fixed bottom-2 right-2 z-40 animate-bounce">
        <NavLink to="/weather">
          {currentUser ? (
            <NavLink to="/chatbot" className="relative inline-block group">
              <img
                src="https://png.pngtree.com/png-clipart/20230401/original/pngtree-smart-chatbot-cartoon-clipart-png-image_9015126.png"
                alt="AI Chatbot"
                className="h-20 w-20 rounded-full border-2 border-green-400 shadow-xl animate-pulse-smooth group-hover:scale-110 transition-transform duration-300"
              />
              <span className="absolute inset-0 rounded-full bg-green-500 opacity-30 blur-lg group-hover:opacity-50 transition duration-300"></span>
            </NavLink>
          ) : (
            <NavLink to="/accesspage" className="relative inline-block group">
              <img
                src="https://png.pngtree.com/png-clipart/20230401/original/pngtree-smart-chatbot-cartoon-clipart-png-image_9015126.png"
                alt="AI Chatbot"
                className="h-20 w-20 rounded-full border-2 border-green-400 shadow-xl animate-pulse-smooth group-hover:scale-110 transition-transform duration-300"
              />
              <span className="absolute inset-0 rounded-full bg-green-500 opacity-30 blur-lg group-hover:opacity-50 transition duration-300"></span>
            </NavLink>
          )}
        </NavLink>
      </div>

      <style>{`
        /* Hide scrollbars globally */
        ::-webkit-scrollbar {
          width: 0;
          height: 0;
        }
        .custom-scrollbar {
          scrollbar-width: none;
          -ms-overflow-style: none;
          scroll-behavior: smooth;
        }
      `}</style>
    </div>


  );
}
