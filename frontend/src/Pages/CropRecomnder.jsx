import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import Header from "../Components/Header";

import {
    Loader,
    Search,
    CloudDrizzle,
    Leaf,
    Thermometer,
    MapPin,
    Droplets,
} from "lucide-react";
import ChatbotIcon from "../Components/ChatbotIcon";

const IconLoader = Loader;
const IconSearch = Search;
const IconCloudDrizzle = CloudDrizzle;
const IconLeaf = Leaf;
const IconThermometer = Thermometer;
const IconMapMarker = MapPin;
const IconWater = Droplets;


export default function CropRecommender() {
    const [place, setPlace] = useState("");
    const [suggestions, setSuggestions] = useState([]);
    const [geocode, setGeocode] = useState(null);
    const [climate, setClimate] = useState(null);
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [remoteCropCatalog, setRemoteCropCatalog] = useState([]);
    const [remoteCropImages, setRemoteCropImages] = useState({});
    const [isCatalogLoaded, setIsCatalogLoaded] = useState(false);

    const HEADERS = {
        "User-Agent": "crop-recommender-app",
    };

    const geocodePlace = async (placeName) => {
        const url = "https://nominatim.openstreetmap.org/search";
        const params = { q: placeName, format: "json", limit: 1, addressdetails: 1 };
        const r = await axios.get(url, { params, headers: HEADERS });
        if (!r.data || r.data.length === 0) return null;
        const row = r.data[0];
        return {
            name: row.display_name,
            lat: parseFloat(row.lat),
            lon: parseFloat(row.lon),
            type: row.type,
            raw: row,
        };
    };

    const fetchClimateNormals = async (lat, lon) => {
        const url = "https://power.larc.nasa.gov/api/temporal/climatology/point";
        const params = {
            latitude: lat,
            longitude: lon,
            community: "AG",
            parameters: "T2M,PRECTOTCORR",
            format: "JSON",
        };
        const r = await axios.get(url, { params });
        const j = r.data;
        const tempDict = j.properties.parameter.T2M;
        const precDict = j.properties.parameter.PRECTOTCORR;

        const monthsOrder = [
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
        ];
        const days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]; // Days in month (approx.)

        const tempMonthly = monthsOrder.map((m) => parseFloat(tempDict[m]));
        const precipMonthly = monthsOrder.map((m, i) => parseFloat(precDict[m]) * days[i]);
        const meanAnnualTemp = tempMonthly.reduce((a, b) => a + b, 0) / 12;
        const annualPrecip = precipMonthly.reduce((a, b) => a + b, 0);

        return { tempMonthly, precipMonthly, meanAnnualTemp, annualPrecip, raw: j };
    };

    const tempMatchScore = (crop, tempC) => {
        const mid = (crop.tMin + crop.tMax) / 2;
        const halfRange = (crop.tMax - crop.tMin) / 2 || 1;
        const score = 1 - Math.abs(tempC - mid) / (halfRange * 2.5);
        return Math.max(0, Math.min(1, score));
    };

    const precipMatchScore = (crop, annualPrecip) => {
        const mid = (crop.pMin + crop.pMax) / 2;
        const halfRange = (crop.pMax - crop.pMin) / 2 || 1;
        const score = 1 - Math.abs(annualPrecip - mid) / (halfRange * 2.5);
        return Math.max(0, Math.min(1, score));
    };

    const monthlySuitability = (crop, climate) => {
        let goodMonths = 0;
        for (let i = 0; i < 12; i++) {
            const t = climate.tempMonthly[i], p = climate.precipMonthly[i];
            const tOk = t >= crop.tMin - 3 && t <= crop.tMax + 3;
            const pOk = p >= crop.pMin / 12 - 10 || p <= crop.pMax / 12 + 50;
            if (tOk && pOk) goodMonths++;
        }
        return goodMonths / 12;
    };

    const scoreCrop = (crop, climate) => {
        const tScore = tempMatchScore(crop, climate.meanAnnualTemp);
        const pScore = precipMatchScore(crop, climate.annualPrecip);
        const seasonScore = monthlySuitability(crop, climate);
        const combined = (0.4 * tScore + 0.4 * pScore + 0.2 * seasonScore);
        return Math.max(0, Math.min(1, combined));
    };

    const generateReason = (c, climateData) => {
        const t = climateData.meanAnnualTemp;
        const p = climateData.annualPrecip;
        let reasons = [];
        reasons.push(`Mean temp ${t.toFixed(1)}°C fits ${c.name}'s preferred ${c.tMin}–${c.tMax}°C`);
        reasons.push(`Annual precip ${Math.round(p)} mm within/near ${c.pMin}–${c.pMax} mm range`);
        if (c.monthsFraction >= 0.7) reasons.push("Many months suitable — could have long growing window");
        else if (c.monthsFraction >= 0.4) reasons.push("Some months suitable — seasonal crop or requires specific sowing times");
        else reasons.push("Limited suitable months — likely requires irrigation/seasonal planning");
        return reasons.join(". ");
    };

    const recommendCrops = useCallback((climateData, topN = 15) => {
        if (!isCatalogLoaded || remoteCropCatalog.length === 0) return [];

        const scored = remoteCropCatalog.map(c => {
            const score = scoreCrop(c, climateData);
            const monthsFraction = monthlySuitability(c, climateData);
            return {
                ...c,
                score,
                monthsFraction,
                image: remoteCropImages[c.name]
            };
        });
        const filtered = scored.filter(c => c.score >= 0.15).sort((a, b) => b.score - a.score);
        return filtered.slice(0, topN).map(c => ({
            name: c.name,
            score: Number(c.score.toFixed(3)),
            image: c.image,
            reason: generateReason({ ...c, monthsFraction: Number(c.monthsFraction.toFixed(2)) }, climateData),
            monthsFraction: Number(c.monthsFraction.toFixed(2)),
            water: c.water || "med"
        }));
    }, [isCatalogLoaded, remoteCropCatalog, remoteCropImages]);

    const fetchForLocation = useCallback(async (lat, lon) => {
        if (!isCatalogLoaded) return;

        setLoading(true); setError("");
        try {
            const climateData = await fetchClimateNormals(lat, lon);
            setClimate(climateData);
            const recs = recommendCrops(climateData, 15);
            setRecommendations(recs);
            const url = "https://nominatim.openstreetmap.org/reverse";
            const params = { lat, lon, format: "json", addressdetails: 1 };
            const r = await axios.get(url, { params, headers: HEADERS });
            setGeocode({ name: r.data.display_name, lat, lon, type: r.data.type || "location" });
        } catch (err) {
            console.error("Fetch error:", err);
            setError("Failed to fetch climate or recommendations.");
        }
        setLoading(false);
    }, [isCatalogLoaded, remoteCropCatalog, recommendCrops]);

    const handleSubmit = async () => {
        setError(""); setGeocode(null); setClimate(null); setRecommendations([]);
        if (!place.trim()) { setError("Please enter an area name."); return; }

        setLoading(true);
        try {
            const geo = await geocodePlace(place);
            if (!geo) {
                setError("Could not geocode the area. Try a more specific name.");
                setLoading(false);
                return;
            }
            setGeocode(geo);
            await fetchForLocation(geo.lat, geo.lon);
        } catch (err) {
            console.error(err);
            setError("An error occurred while fetching data.");
        }
        setLoading(false);
    };

    const fetchSuggestions = async (query) => {
        if (!query.trim()) { setSuggestions([]); return; }
        try {
            const url = "https://nominatim.openstreetmap.org/search";
            const params = { q: query, format: "json", addressdetails: 1, limit: 5 };
            const r = await axios.get(url, { params, headers: HEADERS });
            setSuggestions(r.data || []);
        } catch (err) { console.error(err); setSuggestions([]); }
    };

    useEffect(() => {
        const fetchCatalog = async () => {
            try {
                const response = await axios.get("/api/reccomender/catalog");
                const catalog = response.data.map(c => ({
                    ...c,
                    tMin: parseInt(c.tMin, 10),
                    tMax: parseInt(c.tMax, 10),
                    pMin: parseInt(c.pMin, 10),
                    pMax: parseInt(c.pMax, 10),
                }));

                const images = catalog.reduce((acc, curr) => {
                    acc[curr.name] = curr.imageUrl;
                    return acc;
                }, {});

                setRemoteCropCatalog(catalog);
                setRemoteCropImages(images);
                setIsCatalogLoaded(true);
            } catch (error) {
                console.error("Failed to fetch crop catalog from backend:", error);
                setError("Failed to load crop database. Check server connection.");
                setIsCatalogLoaded(true);
            }
        };
        fetchCatalog();
    }, []);

    // Initial Location Fetch (Runs ONLY after catalog is loaded)
    useEffect(() => {
        if (isCatalogLoaded && remoteCropCatalog.length > 0 && navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                pos => {
                    if (!geocode) {
                        fetchForLocation(pos.coords.latitude, pos.coords.longitude);
                    }
                },
                err => console.warn("Geolocation blocked or unavailable — search manually.", err)
            );
        }
    }, [isCatalogLoaded, fetchForLocation, geocode, remoteCropCatalog.length]);


    const getWaterIcon = (water) => {
        const baseClasses = "flex items-center gap-1 font-medium text-xs rounded-full px-2 py-1 shadow-sm";
        switch (water) {
            case 'high':
                return <span className={`${baseClasses} bg-blue-100 text-blue-800 border border-blue-300`} title="High Water Requirement"><IconWater size={14} /> High Water</span>;
            case 'med':
                return <span className={`${baseClasses} bg-yellow-100 text-yellow-800 border border-yellow-300`} title="Medium Water Requirement"><IconWater size={14} /> Medium Water</span>;
            case 'low':
                return <span className={`${baseClasses} bg-orange-100 text-orange-800 border border-orange-300`} title="Low Water Requirement"><IconWater size={14} /> Low Water</span>;
            default:
                return <span className={`${baseClasses} bg-gray-100 text-gray-800 border border-gray-300`} title="Unknown Water Requirement"><IconWater size={14} /> N/A</span>;
        }
    };

    const getScoreClasses = (score) => {
        if (score >= 0.7) return 'text-green-700 bg-green-100 border-green-300';
        if (score >= 0.4) return 'text-yellow-700 bg-yellow-100 border-yellow-300';
        return 'text-red-700 bg-red-100 border-red-300';
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] text-gray-900 relative overflow-y-auto">
            <Header />

            <div className="bg-gradient-to-t from-green-800 to-green-600 text-white px-4 sm:px-8 lg:px-16 pt-24 pb-8 md:rounded-b-3xl">
                <div className="max-w-6xl mx-auto">
                    <h1 className="text-4xl font-extrabold text-white mb-2 flex items-center gap-2 justify-self-center">
                        <span> Smart Crop Recommender</span>
                    </h1>
                    <p className="text-white mb-0 text-xs md:text-base mx-5 md:justify-self-center pb-3 md:pb-10">
                        Harnessing location-specific climate data for optimized agricultural planning.
                    </p>

                    <div className="bg-white p-4 rounded-xl md:rounded-full shadow-lg border border-gray-200">
                        <div className="flex flex-col sm:flex-row gap-3 relative">
                            <div className="relative flex-1 w-full">
                                <IconSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-green-500 text-lg pointer-events-none" size={20} />
                                <input
                                    type="text"
                                    placeholder="Enter area name, e.g. Ujire, India..."
                                    value={place}
                                    onChange={(e) => { setPlace(e.target.value); fetchSuggestions(e.target.value); }}
                                    className="w-full rounded-full pl-9 pr-3 py-2 bg-gray-100 text-gray-900 placeholder-gray-500 border border-gray-300 focus:border-green-500 focus:ring-1 focus:ring-green-100 outline-none transition-all duration-300 text-sm"
                                />
                                {suggestions.length > 0 && (
                                    <ul className="absolute z-20 w-full bg-white text-gray-900 rounded-lg mt-1 shadow-xl max-h-48 overflow-y-auto border border-gray-300">
                                        {suggestions.map((s, idx) => (
                                            <li key={idx} className="p-2 cursor-pointer hover:bg-green-50 border-b border-gray-100 last:border-b-0 text-xs"
                                                onClick={() => {
                                                    setPlace(s.display_name); setSuggestions([]);
                                                    fetchForLocation(parseFloat(s.lat), parseFloat(s.lon));
                                                }}>
                                                {s.display_name}
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                            <button
                                className="relative overflow-hidden bg-green-600 text-white px-6 py-2 rounded-full shadow-md hover:bg-green-700 transition-all duration-300 font-semibold text-sm disabled:bg-gray-400 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2 group min-w-40"
                                onClick={handleSubmit}
                                disabled={loading || !isCatalogLoaded}
                            >
                                <span className="absolute w-full h-full bg-white opacity-10 transition-transform duration-500 transform -translate-x-full rotate-45 group-hover:translate-x-full"></span>
                                {loading ? (<><IconLoader className="animate-spin" size={20} /> Analyzing...</>) : isCatalogLoaded ? ("Get Recommendation") : (<><IconLoader className="animate-spin" size={20} /> Loading DB...</>)}
                            </button>
                        </div>
                    </div>
                </div>

            </div>

            <main className="w-full max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-0">
                {(loading || !isCatalogLoaded) && (
                    <div className="flex flex-col justify-center items-center py-12">
                        <div className="w-12 h-12 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="text-md font-medium text-green-600 mt-3">
                            {isCatalogLoaded ? "Fetching Geo-location and Climate Data..." : "Loading Crop Database..."}
                        </p>
                    </div>
                )}
                {error && <p className="text-red-600 mt-2 font-medium text-center text-sm">{error}</p>}
                {!loading && isCatalogLoaded && geocode && climate && (
                    <div className="space-y-8">

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                            <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-green-500">
                                <h2 className="text-lg font-bold text-gray-800 mb-2 flex items-center gap-1">
                                    <IconMapMarker className="text-green-500" size={16} /> Location Details
                                </h2>
                                <p className="font-semibold text-gray-700 text-sm">{geocode.name}</p>

                                <div className="mt-2 text-xs text-gray-600 space-y-1">
                                    <p><strong>Lat:</strong> {geocode.lat.toFixed(4)}°</p>
                                    <p><strong>Lon:</strong> {geocode.lon.toFixed(4)}°</p>
                                    <p><strong>Type:</strong> {geocode.type}</p>
                                </div>
                            </div>

                            <div className="lg:col-span-2 bg-white p-4 rounded-lg shadow-md border-l-4 border-blue-500 grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <div>
                                    <h2 className="text-lg font-bold text-gray-800 mb-2 flex items-center gap-1">
                                        <IconThermometer className="text-blue-500" size={16} /> Climate Normals
                                    </h2>
                                    <p className="text-xl font-extrabold text-blue-600">
                                        {climate.meanAnnualTemp.toFixed(2)}°C
                                    </p>
                                    <p className="text-gray-500 text-xs">Mean Annual Temperature</p>
                                </div>

                                <div>
                                    <h2 className="text-lg font-bold text-gray-800 mb-2 flex items-center gap-1">
                                        <IconCloudDrizzle className="text-blue-500" size={16} /> Precipitation
                                    </h2>
                                    <p className="text-xl font-extrabold text-blue-600">
                                        {climate.annualPrecip.toFixed(1)} mm
                                    </p>
                                    <p className="text-gray-500 text-xs">Annual Precipitation</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-white p-5 rounded-xl shadow-lg border border-green-200">
                            <h2 className="text-lg md:text-2xl font-extrabold text-green-700 mb-5 flex items-center gap-2 border-b pb-3 border-green-200">
                                <IconLeaf size={28} /> Top Crop Suitability List
                            </h2>

                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                                {recommendations.map((crop, idx) => (
                                    <div key={idx} className="bg-white rounded-lg shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-200">
                                        <div className="relative">
                                            <img src={crop.image}
                                                alt={crop.name}
                                                className="w-full h-28 object-cover"
                                                onError={(e) => { e.target.onerror = null; e.target.src = "https://placehold.co/400x150/A0E8AF/000?text=No+Image"; }}
                                            />

                                            <div className="absolute top-2 left-2 flex flex-wrap gap-1">
                                                {getWaterIcon(crop.water)}
                                            </div>

                                            <div className={`absolute top-2 right-2 p-1 text-xs font-extrabold rounded-md shadow ${getScoreClasses(crop.score)}`}>
                                                {(crop.score * 100).toFixed(0)}%
                                            </div>
                                        </div>

                                        <div className="p-3">
                                            <h3 className="text-md font-bold text-gray-800 mb-1">{crop.name}</h3>
                                            <p className="text-xs text-gray-600 line-clamp-3 italic">{crop.reason}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {recommendations.length === 0 && (
                                <p className="text-center text-gray-500 p-3 font-semibold text-sm">
                                    No crops met the minimum suitability threshold (15%) or the crop database is empty.
                                </p>
                            )}
                        </div>

                        <div className="bg-white p-5 rounded-xl shadow-lg border border-blue-300">
                            <h2 className="text-base md:text-2xl font-extrabold text-blue-700 mb-4 flex items-center gap-2 border-b pb-3 border-blue-200">
                                <IconThermometer size={28} /> Detailed Monthly Climate Profile
                            </h2>

                            <div className="overflow-x-auto">
                                <table className="min-w-full text-gray-700 border-collapse table-auto text-sm">
                                    <thead className="bg-blue-600 text-white sticky top-0 shadow-md">
                                        <tr>
                                            <th className="px-3 py-2 border border-blue-400 font-semibold text-left rounded-tl-lg">Month</th>
                                            <th className="px-3 py-2 border border-blue-400 font-semibold text-center">Temp (°C)</th>
                                            <th className="px-3 py-2 border border-blue-400 font-semibold text-center rounded-tr-lg">Precip (mm)</th>
                                        </tr>
                                    </thead>

                                    <tbody className="bg-gray-50">
                                        {["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map((m, i) => (
                                            <tr key={m} className="border-b border-gray-200 hover:bg-green-50 transition">
                                                <td className="px-3 py-2 border-r border-gray-200 font-medium">{m}</td>
                                                <td className="px-3 py-2 border-r border-gray-200 text-center font-mono">{climate.tempMonthly[i].toFixed(2)}</td>
                                                <td className="px-3 py-2 text-center font-mono">{climate.precipMonthly[i].toFixed(2)}</td>
                                            </tr>
                                        ))}
                                    </tbody>

                                </table>
                            </div>
                        </div>
                    </div>
                )}
            </main>
            <ChatbotIcon />
        </div>
    );
}