import {
  FaSeedling, FaLeaf, FaExclamationTriangle, FaMountain,
  FaTint, FaSun, FaCalendarAlt, FaFlask, FaChartLine,
  FaThermometerHalf
} from "react-icons/fa";
import { FiSearch } from "react-icons/fi";
import Papa from "papaparse";
import Header from "../Components/Header";
import { useState, useEffect, useCallback } from "react";
import ChatbotIcon from "../Components/ChatbotIcon";

const keyDataIcons = {
  "Ideal pH": FaFlask,
  "Water Needs": FaTint,
  "Sunlight Requirements": FaSun,
  "Soil Type": FaMountain,
  "Planting Season": FaCalendarAlt,
  "Typical Height": FaChartLine,
  "Yield": FaChartLine,
  "Optimal Temperature": FaThermometerHalf,
};

const keyStatHeaders = [
  "Ideal pH",
  "Water Needs",
  "Sunlight Requirements",
  "Soil Type",
  "Planting Season",
  "Typical Height (m)",
  "Yield (Tons/Hectare)",
  "Optimal Temp. (°C)",
];

export default function CropSearchCSV() {
  const [crops, setCrops] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [search, setSearch] = useState("");
  const [result, setResult] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [isFocused, setIsFocused] = useState(false);

  const isInitialState = result?.initial;
  const isErrorState = result?.error;
  const isDataState = result?.["Crop Name"];

  useEffect(() => {
    Papa.parse("../../public/data/Crop_Info_Data.csv", {
      header: true,
      download: true,
      skipEmptyLines: true,
      complete: ({ data, meta }) => {
        const validData = data.filter((row) => row["Crop Name"]?.trim() !== "");
        setCrops(validData);

        if (meta?.fields) {
          const filteredHeaders = meta.fields.filter(
            (key) => key !== "Crop Name" && key !== "Image Description"
          );
          setHeaders(filteredHeaders);
          if (!result) setResult({ initial: true });
        }
      },
    });
  }, [result]);

  const handleSearch = useCallback(
    (query) => {
      const finalQuery = query || search;
      if (!finalQuery) return;

      const foundCrop = crops.find(
        (item) =>
          item["Crop Name"]?.toLowerCase().trim() ===
          finalQuery.toLowerCase().trim()
      );

      setSearch(finalQuery);
      setResult(foundCrop || { error: `No detailed information found for: ${finalQuery}` });
      setSuggestions([]);
      setIsFocused(false);
    },
    [crops, search]
  );

  useEffect(() => {
    if (!search || !isFocused) return setSuggestions([]);

    const match = crops
      .filter((row) =>
        row["Crop Name"]?.toLowerCase().includes(search.toLowerCase())
      )
      .slice(0, 5);

    setSuggestions(match);
  }, [search, crops, isFocused]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSearch(search);
  };

  const handleSuggestionClick = (cropName) => handleSearch(cropName);

  const formatKeyName = (key) => {
    if (key === "Typical Height (m)") return "Typical Height";
    if (key === "Yield (Tons/Hectare)") return "Yield";
    if (key === "Optimal Temp. (°C)") return "Optimal Temperature";
    return key.replace(/([A-Z])/g, " $1").replace(/^./, (x) => x.toUpperCase()).replace(/_/g, " ").trim();
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] text-gray-800 flex flex-col items-center">
      <Header />

      <header className="w-full pt-20 pb-6 bg-gradient-to-t from-green-800 to-green-600 text-white text-center shadow-xl relative overflow-hidden">
        <div className="relative z-10">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight flex justify-center items-center gap-3 drop-shadow-lg">
            Smart Crop Knowledge
          </h1>
          <p className=" text-lime-100 text-sm sm:text-sm mx-5 md:mx-auto mt-4 leading-relaxed">
            Discover vital crop insights from soil and sunlight to yield and optimal conditions
          </p>
        </div>
      </header>

      <main className="w-full px-2 md:px-10 pt-3 md:pt-5 pb-24 space-y-12">
        {(isInitialState || isDataState) && (
          <div className="max-w-6xl mx-auto bg-white/80 backdrop-blur-xl border border-green-200 rounded-3xl shadow-xl p-4 md:p-8 space-y-10">

            <div className="flex flex-col md:flex-row items-center justify-between gap-3 border-b border-green-200 pb-4">
              <div className="flex items-center gap-3">
                <div className="bg-lime-100 p-3 rounded-full shadow-inner">
                  <FaLeaf className="text-green-700 w-6 h-6" />
                </div>
                <h2 className="text-2xl font-extrabold text-green-800">
                  {isDataState ? result["Crop Name"] : "Explore Crops"}
                </h2>
              </div>


              <div className="w-full max-w-lg md:max-w-md relative mt-4 md:mt-0">
                <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-green-600 text-lg" />
                <input
                  type="text"
                  placeholder="Search for a crop..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onFocus={() => setIsFocused(true)}
                  onBlur={() => setTimeout(() => setIsFocused(false), 200)}
                  className="w-full pl-10 pr-28 py-2 bg-green-50/70 border border-green-300 text-green-900 rounded-full placeholder-green-500 outline-none focus:ring-2 focus:ring-green-500 transition"
                />
                <button
                  onClick={() => handleSearch(search)}
                  className="absolute right-1 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-green-600 to-lime-600 hover:from-green-700 hover:to-lime-700 text-white px-4 py-1.5 rounded-full font-semibold text-sm shadow transition"
                >
                  Search
                </button>
                {suggestions.length > 0 && isFocused && (
                  <ul className="absolute left-0 top-full mt-2 w-full bg-white border border-green-200 rounded-xl shadow-lg overflow-hidden max-h-48 overflow-y-auto z-30">
                    {suggestions.map((s, i) => (
                      <li
                        key={i}
                        onClick={() => handleSuggestionClick(s["Crop Name"])}
                        className="px-4 py-2 hover:bg-green-100 cursor-pointer text-sm text-green-800 transition"
                      >
                        {s["Crop Name"]}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

            </div>

            <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
              {headers
                .filter((key) => keyStatHeaders.includes(key))
                .map((key) => {
                  const formatted = formatKeyName(key);
                  const Icon = keyDataIcons[formatted] || FaLeaf;
                  const value = isDataState ? result[key] : (
                    <span className="text-gray-500 italic text-sm">Search a crop to see output</span>
                  );

                  return (
                    <div
                      key={key}
                      className="bg-gradient-to-br from-green-50 to-lime-50 border border-green-200 p-5 rounded-2xl shadow-md hover:shadow-xl hover:scale-105 transition flex flex-col items-center text-center"
                    >
                      <div className="bg-lime-200 p-3 rounded-full mb-2">
                        <Icon className="text-green-700 w-7 h-7" />
                      </div>
                      <h4 className="text-sm font-semibold text-green-700 uppercase">{formatted}</h4>
                      <p className="text-base font-bold text-green-900 mt-1">{value}</p>
                    </div>
                  );
                })}
            </section>

            <section>
              <h3 className="text-xl font-bold text-green-800 mb-4 flex items-center gap-2 pt-10">
                Detailed Cultivation Data
              </h3>

              <div className="overflow-x-auto rounded-2xl border border-green-200 shadow-md bg-white">
                <table className="min-w-full text-sm md:text-base text-green-900 table-fixed">
                  <thead className="bg-green-100">
                    <tr>
                      <th className="text-left px-4 py-3 font-semibold text-green-700 uppercase w-1/3">Parameter</th>
                      <th className="text-left px-4 py-3 font-semibold text-green-700 uppercase w-2/3">Value</th>
                    </tr>
                  </thead>

                  <tbody>
                    {headers
                      .filter((key) => ![
                        "Crop Name",
                        "Image Description",
                        ...keyStatHeaders,
                      ].includes(key))
                      .map((key, i) => {
                        const formatted = formatKeyName(key);
                        const value = isDataState ? result[key] : (
                          <span className="text-gray-500 italic text-sm">Search a crop to see output</span>
                        );

                        return (
                          <tr
                            key={key}
                            className={`border-b border-green-100 ${i % 2 === 0 ? "bg-green-50" : "bg-lime-50"
                              } hover:bg-green-100 transition`}
                          >
                            <td className="px-4 py-4 font-semibold text-green-800 align-top">
                              <div className="flex items-start gap-2">
                                <FaLeaf className="text-lime-600 w-4 h-4 mt-1 flex-shrink-0" />
                                <span>{formatted}</span>
                              </div>
                            </td>

                            <td className="px-4 py-4 text-green-700 font-medium align-top">
                              <div className="min-h-[2.5rem] flex items-start">{value || "—"}</div>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </section>

          </div>
        )}

        {isErrorState && (
          <div className="max-w-3xl mx-auto bg-red-50 border-l-4 border-red-500 text-red-800 p-4 rounded-xl flex items-center gap-3 shadow-md">
            <FaExclamationTriangle className="text-red-600 w-5 h-5" />
            <p className="font-semibold">{result.error}</p>
          </div>
        )}
      </main>

      <ChatbotIcon />
    </div>
  );
}
