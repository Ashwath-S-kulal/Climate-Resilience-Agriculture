import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import { FaSeedling, FaCloudSun, FaVirus, FaLightbulb, FaInfoCircle, FaExclamationTriangle, FaCalendarCheck } from "react-icons/fa";

const images = [
  "https://www.shutterstock.com/image-photo/lush-rice-paddy-field-neat-600nw-2499404003.jpg",
  "https://www.fii.org.in/wp-content/uploads/2020/10/1_Agriculture-1200x480.jpg",
  "https://bsmedia.business-standard.com/_media/bs/img/article/2025-02/07/thumb/featurecrop/1200X630/1738902597-8131.jpg",
  "https://thumbs.dreamstime.com/b/farmer-examining-soil-golden-light-sunset-field-ai-generated-image-farmer-examining-soil-golden-384589200.jpg",
  "https://plasticseurope.org/wp-content/uploads/2021/10/5.6._aaheader.png",
];

const quickActions = [
  {
    to: "/croplist",
    label: "Crops",
    icon: <FaSeedling size={22} />,
    color: "bg-green-500",
    description: "View all crop varieties steps to grow",
  },
  {
    to: "/weather",
    label: "Weather",
    icon: <FaCloudSun size={22} />,
    color: "bg-sky-500",
    description: "Check current weather and forecasts",
  },
  {
    to: "/disease",
    label: "Scan",
    icon: <FaVirus size={22} />,
    color: "bg-amber-500",
    description: "Scan crops for diseases and pests",
  },
  {
    to: "/croprecomnder",
    label: "Recommend",
    icon: <FaLightbulb size={22} />,
    color: "bg-emerald-500",
    description: "Get crop suggestions based on conditions",
  },
  {
    to: "/cropinfo",
    label: "Info",
    icon: <FaInfoCircle size={22} />,
    color: "bg-lime-500",
    description: "Detailed information about each crop",
  },
  {
    to: "/cropriskcalculater",
    label: "Risk",
    icon: <FaExclamationTriangle size={22} />,
    color: "bg-blue-500",
    description: "Calculate potential risks to your crops today",
  },
  {
    to: "/tips",
    label: "Planner",
    icon: <FaCalendarCheck size={22} />,
    color: "bg-orange-500",
    description: "Get tips and plan your farming schedule",
  },
];


export default function WelcomeHero() {
  const [currentSlide, setCurrentSlide] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % images.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + images.length) % images.length);
  };

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % images.length);
  };

  return (

    <div className="bg-white/5 backdrop-blur-lg border border-white/10 rounded-xl shadow-md p-4 m-4">
      <div className="relative w-full h-[55vh] overflow-hidden  border border-green-200 rounded-xl">
        <div
          className="flex transition-transform duration-700 ease-in-out h-full"
          style={{ transform: `translateX(-${currentSlide * 100}%)` }}
        >
          {images.map((img, idx) => (
            <div key={idx} className="flex-shrink-0 w-full h-full relative">
              <img
                src={img}
                alt={`Slide ${idx}`}
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-b from-green-900/60 via-green-800/40 to-green-700/30"></div>
            </div>
          ))}
        </div>

        <button
          onClick={prevSlide}
          className="absolute left-4 top-1/2 transform -translate-y-1/2 text-white bg-black/30 hover:bg-black/50 rounded-full p-3 z-30"
        >
          &#60;
        </button>
        <button
          onClick={nextSlide}
          className="absolute right-4 top-1/2 transform -translate-y-1/2 text-white bg-black/30 hover:bg-black/50 rounded-full p-3 z-30"
        >
          &#62;
        </button>

        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center h-full text-center px-6 pointer-events-none">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold drop-shadow-2xl text-white">
            Welcome Back, <span className="text-green-200">Farmer</span>
          </h1>
          <p className="mt-4 text-base sm:text-lg md:text-2xl max-w-2xl text-green-100/90 leading-relaxed drop-shadow-md">
            Stay ahead of the seasons — monitor your crops, track the weather,
            and optimize your farm management with smart insights.
          </p>
        </div>

        <div className="absolute bottom-4 w-full flex justify-center space-x-2 z-30">
          {images.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setCurrentSlide(idx)}
              className={`w-3 h-3 rounded-full transition-opacity ${idx === currentSlide ? "bg-white opacity-100" : "bg-white opacity-50"
                }`}
            />
          ))}
        </div>
      </div>
      
      <div className="hidden md:block">
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-7 gap-4 text-center mt-10">
          {quickActions.map((item, index) => (
            <NavLink key={index} to={item.to}>
              <div className="group flex flex-col items-center justify-center p-3 bg-white/10 rounded-xl border border-white/10 hover:border-green-300/40 hover:bg-white/20 transition-all duration-300 hover:-translate-y-1 cursor-pointer">
                <div
                  className={`${item.color} w-10 h-10 flex items-center justify-center rounded-full text-white shadow-md group-hover:scale-110 transition-transform duration-300`}
                >
                  {item.icon}
                </div>
                <p className="mt-2 text-sm text-green-100 font-medium group-hover:text-green-200">
                  {item.label}
                </p>
                <p className="mt-1 text-xs text-green-200/80 group-hover:text-green-100">
                  {item.description}
                </p>
              </div>

            </NavLink>
          ))}
        </div>
      </div>
    </div>
  );
}
