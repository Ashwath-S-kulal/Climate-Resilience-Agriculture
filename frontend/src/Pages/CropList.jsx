import React, { useState, useEffect, useRef, useMemo } from "react";
import Papa from "papaparse";
import {
    FiSearch,
    FiRefreshCcw,
    FiX,
    FiEye,
    FiImage,
} from "react-icons/fi";

import { GiTreeGrowth } from "react-icons/gi";

import Header from "../Components/Header";
import CropStepsView from "./CropStepsView";
import ChatbotIcon from "../Components/ChatbotIcon";

export default function CropsList() {
    const [crops, setCrops] = useState([]);
    const [search, setSearch] = useState("");
    const [seasonFilter, setSeasonFilter] = useState("");
    const [selectedCropId, setSelectedCropId] = useState(null);

    const searchInputRef = useRef(null);
    const [showSuggestions, setShowSuggestions] = useState(false);

    useEffect(() => {
        Papa.parse("../../public/data/Crop_Steps_Data.csv", {
            download: true,
            header: true,
            dynamicTyping: true,
            complete: ({ data }) => {
                setCrops(
                    data.filter((d) => d.id && d.name).map((d) => ({
                        ...d,
                        id: Number(d.id),
                    }))
                );
            },
        });
    }, []);

    const selectedCrop = useMemo(
        () => crops.find((c) => c.id === selectedCropId),
        [selectedCropId, crops]
    );

    const searchSuggestions = useMemo(() => {
        if (!search) return [];
        return crops
            .filter((c) => c.name?.toLowerCase().startsWith(search.toLowerCase()))
            .slice(0, 5)
            .map((c) => c.name);
    }, [search, crops]);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (
                searchInputRef.current &&
                !searchInputRef.current.contains(event.target)
            ) {
                setTimeout(() => setShowSuggestions(false), 100);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const filtered = useMemo(() => {
        return crops.filter((c) => {
            const s = search.toLowerCase();
            const matchesSearch = c.name?.toLowerCase().includes(s);
            const matchesSeason = seasonFilter ? String(c.season)?.includes(seasonFilter) : true;
            return matchesSearch && matchesSeason;
        });
    }, [search, seasonFilter, crops]);

    const reset = () => {
        setSelectedCropId(null);
        setSearch("");
        setSeasonFilter("");
        setShowSuggestions(false);
    };

    const handleSuggestionClick = (name) => {
        setSearch(name);
        setSelectedCropId(null);
        setShowSuggestions(false);
    };

    const isDetailViewOpen = selectedCropId !== null;


    return (
        <div className="min-h-screen bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] text-gray-800">
            
            <div className="bg-gradient-to-t from-green-800 to-green-600 text-white md:rounded-b-3xl">
                <Header />
                <div className="pt-24 pb-16 px-4 w-full max-w-7xl mx-auto">
                    <div className="text-center">
                        <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight drop-shadow-lg">
                            Digital Farm Library
                        </h1>
                        <p className="mt-2 text-green-100 drop-shadow">Find detailed guides for your farming needs.</p>
                    </div>
                </div>
            </div>
            
            <div className="px-4 w-full max-w-7xl mx-auto relative -top-9"> 
                <div className="sticky top-16 z-20 bg-white p-4 rounded-xl md:rounded-full shadow-2xl border border-lime-200 mb-8">
                    <div className="flex flex-wrap items-center gap-4">

                        <div className="relative flex-1 min-w-[200px]" ref={searchInputRef}>
                            <div className="flex items-center bg-lime-50 rounded-full px-4 py-2 gap-3 border border-green-300 focus-within:border-green-500 transition-colors">
                                <FiSearch className="text-green-600 text-lg" />
                                <input
                                    type="text"
                                    placeholder="Search crop name..."
                                    className="w-full outline-none text-gray-800 placeholder:text-green-400 bg-transparent"
                                    value={search}
                                    onChange={(e) => {
                                        setSearch(e.target.value);
                                        setSelectedCropId(null);
                                        setShowSuggestions(e.target.value.length > 0);
                                    }}
                                    onFocus={() => search.length >= 1 && setShowSuggestions(true)}
                                />
                                {search && (
                                    <FiX className="text-green-400 cursor-pointer hover:text-red-500" onClick={() => setSearch('')}/>
                                )}
                            </div>
                            {showSuggestions && searchSuggestions.length > 0 && (
                                <div className="absolute w-full mt-2 z-30 max-h-60 overflow-y-auto 
                                    bg-white rounded-lg shadow-xl border border-green-300">
                                    {searchSuggestions.map((name, i) => (
                                        <div
                                            key={i}
                                            className="flex items-center gap-3 px-4 py-2 cursor-pointer 
                                                hover:bg-lime-100 transition-all duration-200"
                                            onMouseDown={(e) => {
                                                e.preventDefault();
                                                handleSuggestionClick(name);
                                            }}
                                        >
                                            <GiTreeGrowth className="text-green-500" />
                                            <span className="font-medium text-gray-700">{name}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div className="flex gap-2">
                            {["Winter", "Spring", "Summer"].map((season) => (
                                <button
                                    key={season}
                                    onClick={() => setSeasonFilter(season === seasonFilter ? "" : season)}
                                    className={`flex items-center text-sm font-semibold px-3 py-2 rounded-full transition-all duration-300 border ${seasonFilter === season
                                        ? "bg-green-600 text-white border-green-600 shadow-md"
                                        : "bg-white hover:bg-lime-50 text-green-600 border-green-300"
                                        }`}
                                >
                                    {season}
                                </button>
                            ))}
                        </div>

                        <button
                            onClick={reset}
                            className="flex items-center gap-2 bg-white hover:bg-red-50 text-green-600 border border-green-300 p-2 px-4 rounded-full shadow-md transition-colors font-medium hover:text-red-600"
                        >
                            <FiRefreshCcw /> Reset
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {filtered.map((c) => (
                        <div
                            key={c.id}
                            onClick={() => setSelectedCropId(c.id)}
                            className={`bg-white rounded-xl overflow-hidden border border-lime-500 shadow-2xl cursor-pointer
                                hover:shadow-green-500/10 hover:scale-[1.03] transition-all duration-500 group relative`}
                        >
                            <div className="h-40 overflow-hidden bg-green-50">
                                {c.image ? (
                                    <img
                                        src={c.image}
                                        alt={c.name}
                                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700 opacity-90"
                                    />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-green-400">
                                        <FiImage className="text-4xl" />
                                    </div>
                                )}
                            </div>

                            <div className="p-4 bg-green-700/90 text-white">
                                <h2 className="text-xl font-bold mb-1 leading-tight text-white">
                                    {c.name}
                                </h2>

                                <div className="flex justify-between items-center mt-3 text-sm">
                                    <span className="flex items-center px-3 py-1 rounded-full text-xs font-semibold 
                                        bg-lime-500/20 text-white border border-lime-500/50">
                                        {c.type}
                                    </span>
                                    <button className="flex items-center text-sm font-semibold text-white hover:text-lime-200 transition-colors duration-300">
                                        View Guide <FiEye className="ml-1 text-base" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                    {filtered.length === 0 && (
                        <div className="col-span-full p-10 text-center bg-white rounded-xl border border-red-300 shadow-xl">
                            <h3 className="text-2xl font-bold text-red-500 mb-2">No Crops Matched</h3>
                            <p className="text-gray-500">Adjust your search or reset the filters.</p>
                        </div>
                    )}
                </div>
            </div>


            <div
                className={`fixed inset-0 z-50 transition-opacity duration-500 
                    ${isDetailViewOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
                onClick={(e) => {
                    if (e.target.classList.contains('backdrop-layer')) {
                        setSelectedCropId(null);
                    }
                }}
            >
                <div className="absolute inset-0 bg-gray-950/80 backdrop-blur backdrop-layer"></div>

                <div
                    className={`absolute bottom-0 left-0 right-0 w-full lg:max-w-4xl max-h-screen lg:max-h-[calc(100vh-80px)] mx-auto 
                        bg-green-700 md:rounded-t-3xl 
                        overflow-scroll flex flex-col transition-transform duration-500 ease-in-out
                        ${isDetailViewOpen ? 'translate-y-0' : 'translate-y-full'}`}
                >
                    {selectedCrop && (
                        <>
                            <div className="p-4 border-b border-green-500 flex justify-between items-center sticky top-0 bg-green-700 z-10">
                                <h2 className="text-2xl font-bold text-lime-200">
                                    {selectedCrop.name} Guide
                                </h2>
                                <button
                                    onClick={() => setSelectedCropId(null)}
                                    className="p-2 bg-lime-400 text-green-900 rounded-full hover:bg-lime-300 transition-colors shadow-lg"
                                >
                                    <FiX className="text-xl" />
                                </button>
                            </div>

                            <div className="h-screen flex-1 overflow-y-auto p-6 custom-scrollbar text-white">
                                <CropStepsView crop={selectedCrop} />
                            </div>
                        </>
                    )}
                </div>
            </div>
            <ChatbotIcon/>
        </div>
    );
}