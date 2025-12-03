import React, { useState, useEffect, useMemo } from "react";
import Papa from "papaparse";
import { FaSyringe, FaFilter, FaSearch, FaChevronDown, FaChevronUp } from "react-icons/fa";
import { IoInformationCircleOutline, IoHeartCircleOutline } from "react-icons/io5";
import Header from "../Components/Header";


export default function CsvReader() {
  const [csvData, setCsvData] = useState([]);
  const [selectedDisease, setSelectedDisease] = useState("");
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);
    fetch("../../public/data/disease_info.csv")
      .then((res) => res.text())
      .then((csv) => {
        Papa.parse(csv, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setCsvData(results.data);
            setIsLoading(false);
          },
        });
      })
      .catch((error) => {
        console.error("Error loading CSV:", error);
        setIsLoading(false);
      });
  }, []);


  const uniqueDiseaseNames = useMemo(() => {
    return [...new Set(csvData.map((row) => row.disease_name).filter(Boolean))].sort();
  }, [csvData]);


  const filteredData = useMemo(() => {
    if (selectedDisease.trim() === "") {
      return csvData;
    }
    return csvData.filter((d) => d.disease_name === selectedDisease);
  }, [csvData, selectedDisease]);



  if (isLoading) {
    return (
      <div className="p-8 min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center p-16 bg-white rounded-2xl shadow-xl max-w-lg border border-indigo-100">
          <p className="text-3xl font-bold text-indigo-600 mb-4 flex items-center justify-center">
            <IoHeartCircleOutline className="mr-3 w-8 h-8 text-teal-500 animate-spin" />
            Loading Disease Data...
          </p>
          <p className="text-gray-500 text-lg">
            Please wait while the dataset is securely retrieved and processed.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-2 md:p-4 min-h-screen bg-gray-50">
      <Header />
      <div className=" mx-auto bg-white shadow-2xl rounded-xl p-3 pt-8 md:p-8 transition-all duration-300  mt-12">
        <div className="text-center mb-10 border-b-4 border-teal-400/50 pb-4 shadow-sm bg-white rounded-xl p-4 pt-0 ">
          <h1 className="md:text-4xl font-extrabold text-indigo-800 flex items-center justify-center space-x-3">
            <FaSyringe className="text-teal-500 w-8 h-8" />
            <span>Disease Data Dashboard</span>
          </h1>
          
        </div>
        <div className=" mx-auto">
          {filteredData.length > 0 ? (
            <div className="bg-white shadow-2xl rounded-2xl overflow-hidden border border-indigo-200">

              <div className="flex flex-col sm:flex-row justify-between items-center bg-gradient-to-t from-green-800 to-green-600 text-white font-bold p-4 sm:p-6">
                <h2 className="text-xl sm:text-2xl font-extrabold flex items-center mb-4 md:mb-0">
                  Disease Entries
                </h2>

                <div className="w-1/2 sm:w-1/3 min-w-[245px] md:min-w-[200px]">
                  <label className="sr-only">Filter by Disease</label>
                  <div className="relative">
                    <FaFilter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-indigo-300 w-4 h-4" />
                    <select
                      className="w-full pl-10 pr-4 py-2 bg-green-600 border border-green-500 rounded-lg shadow-inner text-white font-normal focus:border-teal-400 focus:ring focus:ring-teal-400/50 appearance-none transition cursor-pointer text-sm sm:text-base"
                      value={selectedDisease}
                      onChange={(e) => {
                        setSelectedDisease(e.target.value);
                        setExpandedIndex(null);
                      }}
                    >
                      <option value="" className="text-gray-900 bg-white ">-- View All Diseases --</option>
                      {uniqueDiseaseNames.map((name, idx) => (
                        <option key={idx} value={name} className="text-gray-900 bg-white text-xs md:text-lg">
                          {name}
                        </option>
                      ))}
                    </select>
                    <FaChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white w-3 h-3 pointer-events-none" />
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-[80px_1fr_120px] sm:grid-cols-[100px_1fr_150px] bg-indigo-50 text-indigo-800 font-bold p-4 text-sm sm:text-lg border-b border-indigo-200">
                <div className="hidden sm:block">Image</div>
                <div className="sm:hidden">Pic</div>
                <div>Disease Name</div>
                <div className="text-right">Details</div>
              </div>

              <div className="divide-y divide-gray-100">
                {filteredData.map((row, idx) => {
                  const isExpanded = expandedIndex === idx;
                  return (
                    <div key={idx} className="hover:bg-indigo-50 transition duration-150">
                      <div className="grid grid-cols-[80px_1fr_120px] sm:grid-cols-[100px_1fr_150px] items-center p-4 pl-0 md:p-4">
                        <div className="flex justify-center">
                          {row.image_url ? (
                            <img src={row.image_url} alt={row.disease_name} className="h-12 w-12 sm:h-16 sm:w-16 object-cover rounded-full border-2 border-teal-300 shadow-md" />
                          ) : (
                            <div className="h-12 w-12 sm:h-16 sm:w-16 bg-indigo-100 flex items-center justify-center text-indigo-400 border rounded-full">
                              <IoHeartCircleOutline className="w-5 h-5 sm:w-7 sm:h-7" />
                            </div>
                          )}
                        </div>
                        <div className="text-gray-800 font-semibold text-xs md:text-base">{row.disease_name || "N/A"}</div>

                        <div className="flex justify-end">
                          <button
                            onClick={() => setExpandedIndex(isExpanded ? null : idx)}
                            className={`flex items-center space-x-2 px-1 md:px-3 py-1.5 text-xs md:text-sm font-medium rounded-lg md:rounded-full shadow-lg transition duration-200 
                                            ${isExpanded
                                ? "bg-red-500 text-white hover:bg-red-600"
                                : "bg-teal-500 text-white hover:bg-teal-600"
                              }`
                            }
                          >
                            <span>{isExpanded ? "Hide" : "Know More"}</span>
                            {isExpanded ? <FaChevronUp className="w-3 h-3" /> : <FaChevronDown className="w-3 h-3" />}
                          </button>
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="bg-indigo-50/50 p-6 border-t border-indigo-100 shadow-inner animation-slide-down">

                          <div className="flex flex-col lg:flex-row gap-6">
                            <div className="w-full lg:w-2/5 h-64 lg:h-80 overflow-hidden rounded-xl border-4 border-white shadow-xl bg-white flex-shrink-0">
                              {row.image_url ? (
                                <img
                                  src={row.image_url}
                                  alt={row.disease_name}
                                  className="w-full h-full object-cover transition duration-500 hover:scale-105"
                                />
                              ) : (
                                <div className="flex flex-col items-center justify-center h-full text-indigo-400 text-lg">
                                  <IoHeartCircleOutline className="w-10 h-10 mb-2" />
                                  No Image Available
                                </div>
                              )}
                            </div>

                            <div className="w-full lg:w-3/5">

                              <div className="mb-6 p-4 bg-white rounded-lg shadow border border-gray-100">
                                <h3 className="text-xl font-bold text-indigo-800 mb-2 flex items-center border-b pb-1">
                                  <IoInformationCircleOutline className="mr-2 text-teal-500 w-5 h-5" />
                                  Description
                                </h3>
                                <p className="text-gray-700 leading-relaxed text-sm">
                                  {row.description || <span className="text-gray-500 italic">No description provided.</span>}
                                </p>
                              </div>

                              <div className="p-4 bg-white rounded-lg shadow border border-gray-100">
                                <h3 className="text-xl font-bold text-indigo-800 mb-3 border-b pb-1">Recommended Actions</h3>
                                <ul className="list-disc pl-6 text-gray-700 space-y-2 text-sm">
                                  {(row["Possible Steps"]?.split("\n") || []).filter(step => step.trim() !== "").map((step, i) => (
                                    <li key={i} className="leading-relaxed">{step.trim()}</li>
                                  ))}
                                  {(!row["Possible Steps"] || row["Possible Steps"].trim() === "") && (
                                    <li className="text-gray-500 italic">No specific steps provided. Consult a professional.</li>
                                  )}
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              {filteredData.length === 0 && (
                <div className="text-center p-8 text-gray-500">
                  No data found for the selected disease.
                </div>
              )}
            </div>
          ) : (
            <div className="text-center p-16 bg-white rounded-2xl shadow-xl max-w-lg mx-auto border border-indigo-100 mt-10">
              <p className="text-2xl font-bold text-red-500 mb-2">
                Data Not Available
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}