import { useState, useEffect } from "react";
import axios from "axios";
import Header from "../Components/Header";
import {
  FaLeaf, FaMapMarkerAlt, FaThermometerHalf, FaCloudRain,
  FaExclamationTriangle, FaFire, FaWater, FaCheckCircle,
  FaChartBar, FaSun
} from "react-icons/fa";
import ChatbotIcon from "../Components/ChatbotIcon";


const cropList = [
  "maize",
  "wheat",
  "rice",
  "sorghum",
  "millet",
  "soybean",
  "sugarcane",
  "cotton",
  "groundnut",
  "barley",
  "oats",
  "chickpea",
  "pigeonpea",
  "lentil",
  "mustard",
  "rapeseed",
  "sunflower",
  "potato",
  "tomato",
  "onion",
  "garlic",
  "banana",
  "grapes",
  "apple",
  "orange",
  "tea",
  "coffee",
  "cocoa",
  "rubber",
  "cassava",
  "yam",
  "taro",
  "quinoa",
  "pearl millet",
  "foxtail millet",
  "buckwheat",
  "hemp",
  "tobacco",
  "beetroot",
  "carrot",
  "cabbage",
  "peas",
  "kidney bean",
  "black gram",
  "green gram",
  "cowpea",
  "fenugreek",
  "sesame",
  "jute",
  "flax",
  "rye",
  "triticale",
  "pumpkin",
  "cucumber",
  "bottle gourd",
  "bitter gourd",
  "ridge gourd",
  "brinjal",
  "cauliflower",
  "broccoli",
  "spinach",
  "lettuce",
  "coriander",
  "mint",
  "turmeric",
  "ginger",
  "cardamom",
  "clove",
  "cinnamon",
  "papaya",
  "pineapple",
  "mango",
  "guava",
  "lychee",
  "pomegranate",
  "watermelon",
  "muskmelon",
  "strawberry",
  "blueberry",
  "raspberry",
  "blackberry",
  "almond",
  "walnut",
  "cashew",
  "date palm",
  "fig",
  "olive",
  "avocado",
  "pear",
  "plum",
  "apricot",
  "peach",
  "nectarine",
  "hazelnut",
  "macadamia",
  "cranberry",
  "saffron",
  "vanilla",
  "hops",
  "agave",
  "aloe vera",
  "kale",
  "mustard greens",
  "beet greens",
  "sweet potato",
  "spring onion",
  "leek",
  "asparagus",
  "artichoke",
  "okra",
  "drumstick",
  "tamarind",
  "jackfruit",
  "custard apple",
  "passion fruit",
  "dragon fruit",
  "kiwi",
  "starfruit",
  "longan",
  "durian",
  "breadfruit",
  "plantain"
];

export default function CropRiskCalculater() {
  const [crop, setCrop] = useState("");
  const [place, setPlace] = useState("");
  const [result, setResult] = useState(null);
  const [filteredCrops, setFilteredCrops] = useState([]);
  const [loadingLoc, setLoadingLoc] = useState(false);
  const [loadingRisk, setLoadingRisk] = useState(false);
  const [showCropSug, setShowCropSug] = useState(false);


  useEffect(() => {
    if (!crop) {
      setFilteredCrops([]);
      setShowCropSug(false);
      return;
    }
    const matches = cropList.filter((c) =>
      c.toLowerCase().includes(crop.toLowerCase())
    );
    setFilteredCrops(matches.slice(0, 6));
    setShowCropSug(matches.length > 0);
  }, [crop]);

  const handleUseLocation = async () => {
    if (!navigator.geolocation) {
      alert("Geolocation not supported.");
      return;
    }

    setLoadingLoc(true);
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        try {
          const res = await axios.get(
            `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`
          );
          setPlace(res.data.display_name || `${latitude}, ${longitude}`);
        } catch (error) {
          console.error(error);
          alert("Unable to fetch location name.");
        } finally {
          setLoadingLoc(false);
        }
      },
      (error) => {
        console.error(error);
        alert("Please allow location access.");
        setLoadingLoc(false);
      }
    );
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoadingRisk(true);
    try {
      const res = await axios.post("/api/calculate/riskcalculater", { crop, place });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error calculating risk.");
    } finally {
      setLoadingRisk(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] flex flex-col items-center justify-start md:pb-10">
      <Header />
      <div className="w-full bg-gradient-to-t from-green-800 to-green-600 text-white pb-8 px-4 sm:px-6 shadow-lg pt-20">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-3xl sm:text-4xl font-extrabold mb-3 sm:mb-4 tracking-tight">
            AgriSense <span className="text-green-100">Climate Risk Tool</span>
          </h1>
          <p className="text-green-50 text-sm max-w-2xl mx-auto px-2">
            Understand climate risk for your crops and get tailored recommendations.
          </p>
        </div>
      </div>
      <div className="w-full max-w-6xl px-4 sm:px-6 md:px-8 bg-white/90 backdrop-blur-lg shadow-2xl rounded-none sm:rounded-3xl p-6 sm:p-10 border border-green-100 mt-0 sm:mt-5">
        <div className="grid lg:grid-cols-2 gap-8 lg:gap-10 items-start">
          <div className="p-4 sm:p-6 rounded-2xl border border-green-200 bg-gradient-to-b from-white to-green-50 hover:shadow-xl transition-all h-full">
            <h2 className="text-xl sm:text-2xl font-semibold text-green-700 mb-5 sm:mb-6 flex items-center gap-2">
              <FaCheckCircle className="text-green-500" /> Start finding risks today
            </h2>

            <form onSubmit={submit} className="space-y-5 sm:space-y-6 relative">

              <div className="relative">
                <label className="text-gray-700 font-medium mb-2 flex items-center gap-2 text-sm sm:text-base">
                  <FaLeaf /> Target Crop
                </label>
                <input
                  type="text"
                  value={crop}
                  onChange={(e) => setCrop(e.target.value)}
                  onFocus={() => setShowCropSug(filteredCrops.length > 0)}
                  onBlur={() => setTimeout(() => setShowCropSug(false), 100)}
                  placeholder="e.g., Wheat, Rice, Corn..."
                  className="w-full p-3 rounded-xl border border-gray-300 text-sm sm:text-base focus:ring-2 focus:ring-green-500 outline-none transition"
                  autoComplete="off"
                />
                {showCropSug && filteredCrops.length > 0 && (
                  <div className="absolute top-full left-0 w-full bg-white border rounded-xl shadow-lg max-h-44 overflow-auto z-20">
                    {filteredCrops.map((c, i) => (
                      <div
                        key={i}
                        onMouseDown={() => { setCrop(c); setShowCropSug(false); }}
                        className="p-3 text-sm hover:bg-green-100 cursor-pointer"
                      >
                        {c}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <label className=" text-gray-700 font-medium mb-2 flex items-center gap-2 text-sm sm:text-base">
                  <FaMapMarkerAlt /> Farm Location
                </label>
                <div className="flex flex-col sm:flex-row gap-2 items-stretch sm:items-center">
                  <input
                    type="text"
                    value={place}
                    onChange={(e) => setPlace(e.target.value)}
                    placeholder="e.g., Delhi, India"
                    className="flex-1 w-full p-3 rounded-xl border border-gray-300 text-sm sm:text-base focus:ring-2 focus:ring-green-500 outline-none transition"
                    autoComplete="off"
                  />
                  <button
                    type="button"
                    onClick={handleUseLocation}
                    disabled={loadingLoc}
                    className="bg-green-600 hover:bg-green-700 text-white px-4 p-3 rounded-xl transition flex items-center gap-2 justify-center text-sm w-full sm:w-auto min-w-[160px]"
                  >
                    {loadingLoc ? (
                      <>
                        <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                        Fetching...
                      </>
                    ) : (
                      <div className="flex gap-2 items-center">
                        <FaMapMarkerAlt /> Use Current Location
                      </div>
                    )}
                  </button>
                </div>
              </div>
              <button
                type="submit"
                disabled={loadingRisk}
                className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-bold py-3 rounded-xl transition flex justify-center items-center gap-2 text-base sm:text-lg"
              >
                {loadingRisk ? (
                  <>
                    <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                    Calculating...
                  </>
                ) : (
                  <>
                    <FaCheckCircle /> Check Climate Risk
                  </>
                )}
              </button>
            </form>
          </div>
          <div className="p-4 sm:p-6 bg-gradient-to-b from-white to-green-50 rounded-2xl border border-dashed border-green-300 hover:shadow-xl transition-all min-h-[400px] sm:min-h-[450px] flex flex-col justify-center">
            {loadingRisk ? (
              <div className="flex flex-col justify-center items-center h-full space-y-4">
                <span className="w-12 h-12 border-4 border-green-600 border-t-transparent rounded-full animate-spin"></span>
                <p className="text-green-700 font-medium text-lg">Calculating Climate Risks...</p>
              </div>
            ) : (
              <>
                <h2 className="text-xl sm:text-2xl font-semibold text-green-700 mb-5 sm:mb-6 flex items-center gap-2">
                  <FaChartBar /> Climate Risk Results
                </h2>
                <div className="space-y-3 sm:space-y-4 text-left text-sm sm:text-base">
                  <p className="flex items-start gap-2"><FaMapMarkerAlt className="text-green-600 mt-1" /><span className="font-bold">Location:</span>
                  <span className="pl-2 whitespace-pre-line break-words">
                      {result?.location?.place || (
                        <span className="text-sm italic text-gray-600">
                          Awaiting input...
                        </span>
                      )}
                    </span>
                  </p>
                  <p><FaThermometerHalf className="inline mr-2 text-orange-600" /><b>Temperature:</b> {result?.weatherData?.temperature !== undefined ? `${result.weatherData.temperature}°C` : <span className="text-sm italic text-gray-600">Awaiting input...</span>}</p>
                  <p><FaCloudRain className="inline mr-2 text-blue-600" /><b>Rainfall:</b> {result?.weatherData?.rainfall !== undefined ? `${result.weatherData.rainfall} mm` : <span className="text-sm italic text-gray-600">Awaiting input...</span>}</p>
                </div>
                <div className="mt-6 sm:mt-8">
                  <h3 className="font-semibold text-green-700 mb-3 sm:mb-4 flex items-center gap-2 text-base sm:text-lg"><FaExclamationTriangle /> Risk Scores</h3>
                  <div className="grid grid-cols-3 gap-3 sm:gap-4 text-center">
                    <div className="bg-red-50 p-3 sm:p-4 rounded-xl border border-red-200">
                      <FaFire className="text-red-500 mb-1 text-base sm:text-lg" />
                      <p className="font-medium text-gray-700 text-xs sm:text-sm">Drought</p>
                      <p className="text-lg sm:text-xl font-bold text-red-600">{result?.riskScores?.droughtRisk.toFixed(2) ?? "*"}/5</p>
                    </div>
                    <div className="bg-blue-50 p-3 sm:p-4 rounded-xl border border-blue-200">
                      <FaWater className="text-blue-500 mb-1 text-base sm:text-lg" />
                      <p className="font-medium text-gray-700 text-xs sm:text-sm">Flood</p>
                      <p className="text-lg sm:text-xl font-bold text-blue-600">{result?.riskScores?.floodRisk.toFixed(2) ?? "*"}/5</p>
                    </div>
                    <div className="bg-orange-50 p-3 sm:p-4 rounded-xl border border-orange-200">
                      <FaSun className="text-orange-500 mb-1 text-base sm:text-lg" />
                      <p className="font-medium text-gray-700 text-xs sm:text-sm">Heat</p>
                      <p className="text-lg sm:text-xl font-bold text-orange-600">{result?.riskScores?.heatRisk.toFixed(2) ?? "*"}/5</p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
        <div className="mt-8 bg-gray-300/50 shadow-2xl w-full py-6 sm:py-10 px-4 sm:px-8 rounded-2xl">
          <h3 className="text-xl sm:text-2xl font-bold text-green-700 mb-3 flex items-center gap-2 justify-self-center"><FaLeaf /> Recommendations</h3>
          <ul className="list-none ml-0 space-y-2 text-gray-700">
            {result?.recommendations?.length > 0
              ? result.recommendations.map((r, i) => (
                <li key={i} className="p-3 bg-green-50 rounded-lg hover:bg-green-100 transition flex items-start gap-2 text-sm sm:text-base">
                  <FaCheckCircle className="text-green-500 mt-1 flex-shrink-0" /> <span>{r}</span>
                </li>
              ))
              : <span className="text-sm italic text-gray-600 flex justify-self-center p-3">No recommendations yet.</span>
            }
          </ul>
        </div>
      </div>

      <ChatbotIcon />
    </div>
  );
}
