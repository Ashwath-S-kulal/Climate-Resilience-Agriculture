import React, { useState, useEffect, useMemo } from "react";
import Papa from "papaparse";
import { FiChevronDown, FiChevronUp, FiExternalLink } from "react-icons/fi";
import { FaChevronDown, FaFilter, FaSyringe } from "react-icons/fa";
import Header from "../Components/Header";

export default function CsvReader() {
  const [csvData, setCsvData] = useState([]);
  const [expandedRow, setExpandedRow] = useState(null);
  const [selectedDisease, setSelectedDisease] = useState("");
  const [allData, setAllData] = useState([]);

  useEffect(() => {
    fetch("../../public/data/supplement_info.csv")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.text();
      })
      .then((csv) => {
        Papa.parse(csv, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const parsedData = results.data.map(row => ({
              ...row,
              disease_name: row.disease_name && row.disease_name.trim() !== ""
                ? row.disease_name.trim()
                : "General Wellness"
            }));
            setAllData(parsedData);
            setCsvData(parsedData);
          },
        });
      })
      .catch((error) => console.error("Error loading CSV:", error));
  }, []);

  const handleFilterChange = (event) => {
    const disease = event.target.value;
    setSelectedDisease(disease);

    if (disease === "") {
      setCsvData(allData);
    } else {
      const filteredData = allData.filter(
        (row) => row.disease_name === disease
      );
      setCsvData(filteredData);
    }
    setExpandedRow(null);
  };


  const uniqueDiseases = useMemo(() => {
    const diseases = allData.map(row => row.disease_name);
    return [...new Set(diseases)].sort();
  }, [allData]);

  const toggleRow = (index) => {
    setExpandedRow(expandedRow === index ? null : index);
  };

  return (
    <div className=" md:p-8 min-h-screen bg-gradient-to-br from-teal-50 to-cyan-100 font-sans mt-14">
      <Header />
      <div className=" mx-auto bg-white shadow-2xl rounded-xl p-3 md:p-8 transition-all duration-300 ">
        <div className="text-center mb-10 border-b-4 border-teal-400/50 pb-4 shadow-sm bg-white rounded-xl p-4">
          <h1 className="text-xl md:text-4xl font-extrabold text-indigo-800 flex items-center justify-center space-x-3">
            <FaSyringe className="text-teal-500 w-8 h-8" />
            <span>Supplement Product Catalog</span>
          </h1>
          <p className="text-gray-500 mt-2 text-lg">
          </p>
        </div>
        <h1 className="text-4xl font-extrabold mb-10 text-center text-teal-800 tracking-wider">
        </h1>
        <div className="flex flex-col sm:flex-row justify-between items-center bg-indigo-700 text-white font-bold p-4 sm:p-6 gap-4 rounded-t-xl">
          <h2 className="text-xl sm:text-2xl font-extrabold flex items-center">
            Disease Entries
          </h2>

          <div className="relative w-full sm:w-1/3 min-w-[220px]">
            <label htmlFor="disease-filter" className="sr-only">Filter by Disease</label>
            <FaFilter className="absolute left-3 top-1/2 -translate-y-1/2 text-indigo-300 w-4 h-4" />
            <select
              id="disease-filter"
              value={selectedDisease}
              onChange={handleFilterChange}
              className="pl-10 pr-8 py-2 text-black text-xs md:text-base bg-white border border-indigo-300 rounded-lg shadow-sm 
                focus:ring-indigo-500 focus:border-indigo-500 w-full appearance-none"
            >
              <option value="">All Diseases</option>
              {uniqueDiseases.map((disease) => (
                <option key={disease} value={disease}>
                  {disease}
                </option>
              ))}
            </select>
            <FaChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-indigo-600 w-3 h-3 pointer-events-none" />
          </div>
        </div>


        {csvData.length === 0 && allData.length > 0 ? (
          <div className="text-center p-10 text-gray-500 text-xl">
            No supplements found for the selected disease. Try a different filter or reset.
          </div>
        ) : (
          <div className="w-full overflow-x-auto rounded-xl rounded-t-none">
            <table className="w-full border-collapse overflow-hidden min-w-[600px]">
              <thead>
                <tr className="bg-teal-600 text-white font-extrabold text-xs md:text-base tracking-wider shadow-lg">
                  <th className="p-2 md:p-4 border-r border-teal-500 w-fit md:w-[10%]">
                    Product Image
                  </th>
                  <th className="p-2 md:p-4 border-r border-teal-500 w-fit md:w-[70%] text-left">
                    Recommended Supplement For
                  </th>
                  <th className="p-2 md:p-4 w-[20%]">Details</th>
                </tr>
              </thead>

              <tbody>
                {csvData.map((row, idx) => (
                  <React.Fragment key={idx}>
                    <tr className="bg-white even:bg-teal-50 hover:bg-teal-100 transition duration-200 border-teal-500 border-2 border-b-0">
                      <td className="p-3 border-r border-teal-200 text-center">
                        {row["supplement image"] ? (
                          <img
                            src={row["supplement image"]}
                            alt={row["supplement name"] || "Supplement"}
                            className="h-12 w-12 md:h-16 md:w-16 object-contain mx-auto rounded-md shadow-sm border border-gray-200"
                          />
                        ) : (
                          <span className="text-gray-400 text-xs">No Image</span>
                        )}
                      </td>

                      <td className="p-2 md:p-4 border-r border-teal-200 text-left text-gray-700 text-xs md:text-base break-words">
                        <span className="text-teal-800 font-bold">
                          Recommended Supplement For:
                          <span className="text-green-600 font-mono"> {row.disease_name}</span>
                        </span>
                        <br />
                      </td>

                      <td className="md:p-3 text-center">
                        <button
                          onClick={() => toggleRow(idx)}
                          className={`flex items-center justify-center gap-2 px-4 py-2 w-28 md:w-32 mx-auto rounded-full font-semibold shadow-lg transition duration-200 transform ${expandedRow === idx
                              ? "bg-red-500 text-white hover:bg-red-600 hover:scale-105"
                              : "bg-teal-500 text-white hover:bg-teal-600 hover:scale-105"
                            }`}
                        >
                          {expandedRow === idx ? (
                            <>
                              <FiChevronUp className="text-lg" /> Hide
                            </>
                          ) : (
                            <>
                              <FiChevronDown className="text-lg" /> View
                            </>
                          )}
                        </button>
                      </td>
                    </tr>

                    {expandedRow === idx && (
                      <tr>
                        <td
                          colSpan={3}
                          className="border-2 border-teal-400 bg-gray-50 p-4 md:p-6 border-t-0"
                        >
                          <div className="flex flex-col md:flex-row items-start gap-6 md:gap-8">
                            <div className="w-full md:w-1/4 flex justify-center p-4 bg-white rounded-lg shadow-inner">
                              {row["supplement image"] ? (
                                <img
                                  src={row["supplement image"]}
                                  alt={row["supplement name"] || "Supplement"}
                                  className="max-h-40 md:max-h-52 object-contain rounded-lg transition-all duration-300"
                                />
                              ) : (
                                <div className="h-40 w-full flex items-center justify-center bg-gray-100 text-gray-500 rounded-lg">
                                  No Image Available
                                </div>
                              )}
                            </div>

                            <div className="w-full md:w-3/4">
                              <h2 className="text-xl md:text-2xl font-extrabold text-teal-800 mb-2 border-b-2 border-teal-300 pb-1">
                                {row["supplement name"] || "Product Details"}
                              </h2>

                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 text-gray-700 mb-4">
                                <p>
                                  <strong className="text-teal-700">Disease Targeted:</strong>{" "}
                                  {row.disease_name}
                                </p>
                                <p>
                                  <strong className="text-teal-700">Product ID:</strong>{" "}
                                  {row.index || "N/A"}
                                </p>
                              </div>

                              {row["buy link"] ? (
                                <a
                                  href={row["buy link"]}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-2 px-5 py-3 bg-green-600 text-white font-bold rounded-full shadow-xl mt-3 hover:bg-green-700 transition duration-300 transform hover:translate-y-[-2px]"
                                >
                                  <FiExternalLink className="text-xl" /> View & Buy Now
                                </a>
                              ) : (
                                <span className="text-red-500 font-semibold mt-3 block">
                                  Purchase Link Unavailable
                                </span>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>

        )}
      </div>
    </div>
  );
}