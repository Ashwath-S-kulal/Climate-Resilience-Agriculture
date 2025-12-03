import React, { useState, useEffect, useRef } from "react";
import Header from "../Components/Header";
import axios from "axios";
import ChatbotIcon from "../Components/ChatbotIcon";
import { NavLink } from "react-router-dom";
import { useSelector } from "react-redux";
import { ImageOff, RefreshCw, Trash2 } from "lucide-react";


const blobToFile = (blob, fileName) => {
  return new File([blob], fileName, { type: blob.type });
};

const CameraView = ({ onCapture, onClose }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let stream;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" } })
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            setIsReady(true);
          }
        }
      })
      .catch((err) => {
        console.error("Error accessing camera: ", err);
        alert("Could not access camera. Please ensure permissions are granted.");
        onClose();
      });

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [onClose]);

  const handleCapture = () => {
    if (!isReady || !videoRef.current || !canvasRef.current) return;

    const context = canvasRef.current.getContext("2d");
    const video = videoRef.current;

    canvasRef.current.width = video.videoWidth;
    canvasRef.current.height = video.videoHeight;
    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    canvasRef.current.toBlob((blob) => {
      if (blob) {
        const capturedFile = blobToFile(blob, `capture-${Date.now()}.jpg`);
        onCapture(capturedFile);
      } else {
        alert("Failed to capture image.");
      }
    }, "image/jpeg");
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-full h-auto max-h-[200px] rounded-lg border border-emerald-400 shadow-inner object-cover"
      ></video>
      <canvas ref={canvasRef} className="hidden"></canvas>

      <div className="flex justify-center space-x-3 w-full">
        <button
          onClick={onClose}
          type="button"
          className="flex-1 py-2 text-sm font-semibold text-red-600 border border-red-300 rounded-xl hover:bg-red-50 transition-all"
        >
          Cancel
        </button>
        <button
          onClick={handleCapture}
          disabled={!isReady}
          type="button"
          className="flex-1 py-2 text-sm font-semibold bg-gradient-to-r from-emerald-500 to-green-500 text-white rounded-xl hover:from-emerald-600 hover:to-green-600 transition-all disabled:opacity-50"
        >
          Capture
        </button>
      </div>
    </div>
  );
};

export default function Disease() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const [uploadedImageFile, setUploadedImageFile] = useState(null);
  const [capturedImageFile, setCapturedImageFile] = useState(null);
  const { currentUser } = useSelector(state => state.user);



  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`/api/predictions/getPredictions/${currentUser._id}`);
      setHistory(res.data);
    } catch (err) {
      console.error("Error fetching history:", err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // Sync the primary 'image' state with the currently selected input (uploaded or captured)
  useEffect(() => {
    if (capturedImageFile) {
      setImage(capturedImageFile);
      setUploadedImageFile(null);
    } else if (uploadedImageFile) {
      setImage(uploadedImageFile);
      setCapturedImageFile(null);
    } else {
      setImage(null);
    }
  }, [uploadedImageFile, capturedImageFile]);

  const handleSubmit = async (e) => {
    setLoading(true);
    e.preventDefault();
    if (!image) 
      return alert("Please upload or capture an image!");

    const formData = new FormData();
    formData.append("image", image);
    try {
      const res = await axios.post('/api/predictions/predict', formData, {
  headers: { "Content-Type": "multipart/form-data" },
});


      const predictionResult = res.data;
      if (predictionResult.confidence < 0.15) {
        setResult({ invalid: true });   
      } else {
        setResult(predictionResult); 
        await axios.post(`/api/predictions/createPrediction/${currentUser._id}`, predictionResult);
        fetchHistory();
      }
    } catch (err) {
      console.error("Prediction error:", err);
      alert("Failed to predict disease. Check backend server.");
    }
    setLoading(false);
  };


  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this record?")) return;
    try {
      await axios.delete(`/api/predictions/deletePrediction/${id}`);
      fetchHistory();
    } catch (err) {
      console.error("Error deleting prediction:", err);
      alert("Failed to delete prediction.");
    }
  };

  const handleCaptureImage = (file) => {
    setCapturedImageFile(file);
    setIsCameraActive(false);
  };

  const handleUploadImage = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedImageFile(file);
      setCapturedImageFile(null);
    }
  };

  const clearUploadedImage = () => {
    setUploadedImageFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  const clearCapturedImage = () => {
    setCapturedImageFile(null);
    setIsCameraActive(false);
  }

  const handleClearHistory = async (userId) => {
    if (!window.confirm("Are you sure you want to clear all prediction history?")) return;
    try {
      await axios.delete(`/api/predictions/clearPrediction/${userId}`);
      fetchHistory();
    } catch (err) {
      console.error("Error clearing history:", err);
    }
  };



  return (
    <div className="min-h-screen bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] text-gray-800 relative overflow-y-auto pt-14 ">
      <Header />
      <div className={`flex-1  gap-6 flex flex-col transition-all duration-300 `}>
        <div className="w-full bg-gradient-to-t from-green-800 to-green-600 text-white py-9 px-6 shadow-lg rounded-b-3xl">
          <div className="max-w-5xl mx-auto text-center">
            <h1 className="text-2xl md:text-4xl font-extrabold mb-4 tracking-tight">
              Plant Disease Detection
            </h1>
            <p className="text-green-50 text-xs md:text-md max-w-2xl mx-auto">
              Upload or capture a leaf image to analyze its health condition.
            </p>
          </div>
        </div>

        <form
          onSubmit={handleSubmit}
          className=" bg-white  rounded-xl border border-gray-200 space-y-6 mx-8 pb-0 my-0"
        >
          <div className="bg-emerald-600 rounded-t-xl p-4 flex flex-col items-center">
            <h2 className="text-xl font-bold text-white">
              Plant Disease Detection
            </h2>
            <p className="text-emerald-100/90 text-sm">
              Upload or capture a leaf image to analyze its health condition
            </p>
          </div>

          <div className="p-4 space-y-6">
            <div className="flex space-x-4">
              <div className="flex-1 flex flex-col items-center justify-center bg-gray-100 p-4 border-2 border-dashed border-gray-300 rounded-xl hover:border-emerald-400 transition-all min-h-[140px] relative shadow-inner">
                {uploadedImageFile ? (
                  <div className="flex flex-col items-center justify-center space-y-2">
                    <p className="font-medium text-sm text-center truncate max-w-[120px]">
                      {uploadedImageFile.name}
                    </p>
                    <img
                      src={URL.createObjectURL(uploadedImageFile)}
                      alt="uploaded preview"
                      className="w-16 h-16 object-cover rounded-md border border-gray-300 shadow-md"
                    />
                    <button
                      type="button"
                      onClick={clearUploadedImage}
                      className="text-red-600 hover:text-red-500 text-xs font-semibold"
                    >
                      Change
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => fileInputRef.current.click()}
                    className="flex flex-col items-center justify-center h-full w-full p-2 text-gray-600 hover:text-emerald-600"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-8 h-8 mb-2"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M3 16l4-4a2 2 0 012.828 0L15 17m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <p className="font-medium text-sm text-center">
                      Upload File
                    </p>
                    <p className="text-xs mt-0.5 text-gray-500">
                      PNG, JPG, JPEG
                    </p>
                  </button>
                )}

                {/* Hidden File Input */}
                <input
                  ref={fileInputRef}
                  id="fileInput"
                  type="file"
                  accept="image/*"
                  onChange={handleUploadImage}
                  className="hidden"
                />
              </div>

              <div className="flex-1 flex flex-col items-center justify-center p-4 bg-gray-100 rounded-xl min-h-[140px] border-2 border-dashed border-gray-300 ">
                {isCameraActive ? (
                  <CameraView
                    onCapture={handleCaptureImage}
                    onClose={clearCapturedImage}
                  />
                ) : capturedImageFile ? (
                  <div className="flex flex-col items-center justify-center space-y-2">
                    <p className="font-medium text-sm text-center">
                      Image Captured
                    </p>
                    <img
                      src={URL.createObjectURL(capturedImageFile)}
                      alt="captured preview"
                      className="w-16 h-16 object-cover rounded-md border border-gray-300 shadow-md"
                    />
                    <button
                      type="button"
                      onClick={clearCapturedImage}
                      className="text-red-600 hover:text-red-500 text-xs font-semibold"
                    >
                      Change
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      setUploadedImageFile(null);
                      setIsCameraActive(true);
                    }}
                    className="flex flex-col items-center justify-center h-full w-full p-2 text-gray-600 hover:bg-emerald-50 hover:text-emerald-600 rounded-lg transition-all"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-8 h-8 mb-2"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.218A2 2 0 0110.207 4h3.586a2 2 0 011.664.89l.812 1.218A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                    </svg>
                    <p className="font-medium text-sm text-center">
                      Capture from Camera
                    </p>
                    <p className="text-xs mt-0.5 text-gray-500">
                      Real-time
                    </p>
                  </button>
                )}
              </div>
            </div>

            <button
              type="submit"
              disabled={isCameraActive || !image}
              className="w-full bg-gradient-to-r from-emerald-500 to-green-500 text-white py-3 rounded-xl font-semibold hover:from-emerald-600 hover:to-green-600 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 flex items-center justify-center gap-2 mt-auto"
            >
              ⚡ Analyze Disease
            </button>
          </div>
          <div className="w-full bg-white p-0 shadow-2xl border-2  border-gray-200 border-t-2 border-t-blue-400 min-h-[317px] rounded-b-xl">
            <div className="p-4 pb-0 flex justify-center ">
              <h3 className="text-xl font-bold text-black border-b-4 border-b-green-700">
                Analysis Result
              </h3>
            </div>
            {(loading) && (
              <div className="flex flex-col justify-center items-center py-12">
                <div className="w-8 h-8 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                <p className="text-md font-medium text-green-600 mt-3">
                  Predicting disease...
                </p>
              </div>
            )}
            {!loading && (
              <div>
                <div className="h-[250px] p-6 rounded-xl m-2 bg-white shadow-md border border-green-200">
                  {!result ? (
                    <div className="h-full flex items-center justify-center text-gray-500 text-sm italic py-8">
                      Select an image and analyze to view results
                    </div>
                  ) : result.invalid ? (
                    <div className="h-full flex flex-col items-center justify-center text-red-600 text-lg font-bold py-8">
                      <ImageOff size={48} className="text-red-500" />Not a valid image
                      <p className="text-sm text-gray-600 mt-2">Please upload a clear leaf image.</p>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-4 text-lg">
                      <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                        <p className="font-semibold text-green-800">Prediction:</p>
                        <p className="text-xl font-bold text-green-700 mt-1">{result.prediction}</p>
                      </div>

                      <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                        <p className="font-semibold text-green-800">Confidence:</p>
                        <p className="text-xl font-bold text-green-700 mt-1">{(result.confidence * 100).toFixed(2)}%</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </form>
        <div className=" mt-4 p-6 m-8 bg-white rounded-xl shadow-2xl border border-gray-100">
          <h3 className="text-xl font-extrabold text-green-700 mb-4 border-b-4 border-green-500 pb-2 inline-block">
            Quick Navigation
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <NavLink to={"/diseasedata"} className="block transform hover:scale-[1.02] transition-transform duration-300">
              <div className="bg-gray-50 p-6 rounded-2xl border-2 border-green-300 shadow-xl hover:shadow-green-400/50 transition-shadow duration-300 h-full flex flex-col cursor-pointer">
                <div className="flex items-center space-x-4 mb-3">
                  <div className="bg-green-500 p-3 rounded-full shadow-md">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="white" className="w-6 h-6">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125l7.5-7.5 7.5 7.5-7.5 7.5M12 9l3 3m0 0l-3 3m0-6l-3 3" />
                    </svg>
                  </div>
                  <h4 className="text-xl font-bold text-green-800">
                    Disease Data Dashboard
                  </h4>
                </div>
                <p className="text-base text-gray-700 mt-2">
                  Explore comprehensive information on various health conditions, their Description and Steps to Reduce that crop diseases.
                </p>
              </div>
            </NavLink>

            <NavLink to={"/diseasesuppliment"} className="block transform hover:scale-[1.02] transition-transform duration-300">
              <div className="bg-gray-50 p-6 rounded-2xl border-2 border-green-300 shadow-xl hover:shadow-green-400/50 transition-shadow duration-300 h-full flex flex-col cursor-pointer">
                <div className="flex items-center space-x-4 mb-3">
                  <div className="bg-green-500 p-3 rounded-full shadow-md">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="white" className="w-6 h-6">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 7.5l-.64 9.143a2 2 0 01-1.996 1.857H6.38a2 2 0 01-1.996-1.857L3.75 7.5m1.5-1.5h15l-1.5-3h-12l-1.5 3zm6 4.5v-3h3v3h-3z" />
                    </svg>
                  </div>
                  <h4 className="text-xl font-bold text-green-800">
                    Supplement Product Catalog
                  </h4>
                </div>
                <p className="text-base text-gray-700 mt-2">
                  View and order suggested treatment products, fertilizers, and supplements based on analysis recommendations.
                </p>
              </div>
            </NavLink>
          </div>


          <div
            className="mt-4 rounded-xl bg-gray-50 border-2 border-green-300 shadow-xl hover:shadow-green-400/50 transition-shadow duration-300"
          >
            <div className="bg-emerald-600 p-4 sticky top-0 z-10 rounded-t-xl flex justify-between items-center">
              <h3 className="text-xl font-bold text-white">
                Your Prediction History
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={fetchHistory}
                  className="px-3 py-1 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-all shadow flex items-center gap-1"
                >
                  <RefreshCw size={16} />
                  Refresh History
                </button>
                <button
                  onClick={() => handleClearHistory(currentUser._id)}
                  className="px-3 py-1 bg-red-500 text-white text-sm rounded-lg hover:bg-red-600 transition-all shadow flex items-center gap-1"
                >
                  <Trash2 size={16} />
                  Clear history
                </button>
              </div>
            </div>
            <div className="p-4 sm:p-6 flex flex-col gap-4">
              {(loading) && (
                <div className="flex flex-col justify-center items-center py-12">
                  <div className="w-8 h-8 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-md font-medium text-green-600 mt-3">
                    Refreshing history...
                  </p>
                </div>
              )}
              {!loading && history.length > 0 ? (
                history.map((h) => (
                  <div
                    key={h._id}
                    className="bg-white p-4 sm:p-5 rounded-xl border border-gray-200 hover:bg-gray-100 transition-all shadow-md flex flex-col gap-3"
                  >
                    <p className="text-gray-800 font-bold break-words text-xs md:text-sm sm:text-base">
                      Disease Name :
                      <span className="text-green-800 ml-1">{h.prediction}</span>
                    </p>

                    <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center text-xs sm:text-sm text-gray-600 gap-3">
                      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-8">
                        <span>
                          <strong>Disease Confidence:</strong>{" "}
                          {(h.confidence * 100).toFixed(2)}%
                        </span>

                        <span>
                          <strong>Predicted Date:</strong>{" "}
                          {new Date(h.date).toLocaleDateString()}
                        </span>
                      </div>
                      <button
                        onClick={() => handleDelete(h._id)}
                        className="text-red-600 hover:text-red-700 font-bold transition flex gap-1 items-center"
                      >
                        <Trash2 size={18} />
                        Delete
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div>
                  {!loading && (
                    <p className="text-gray-500 text-center mt-8 text-sm sm:text-base">
                      No predictions yet.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <ChatbotIcon />
    </div>
  );
}