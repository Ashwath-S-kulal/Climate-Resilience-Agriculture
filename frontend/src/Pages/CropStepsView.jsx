import React, { useEffect, useState } from "react";
import { FiClipboard, FiZap } from "react-icons/fi";

export default function CropStepsView({ crop }) {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!crop?.num) return;

    setLoading(true);

    fetch(`/api/cropsteps/${crop.num}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch crop steps");
        return res.json();
      })
      .then((data) => {
        setPlan({
          overview: data.overview || "",
          steps: Array.isArray(data.steps) ? data.steps : []
        });
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading crop steps:", err);
        setPlan(null);
        setLoading(false);
      });
  }, [crop?.num]);

  if (loading)
    return <p className="text-center mt-10 text-gray-600">Loading...</p>;

  if (!plan)
    return (
      <p className="text-white italic text-center mt-10 p-2 rounded-lg bg-green-700/50 text-sm">
        No growing data available for {crop.name} yet.
      </p>
    );

  const { overview, steps } = plan;

  return (
    <div className="w-full h-full bg-[#F0FAF2] border border-green-200 rounded-2xl shadow-lg p-6 md:p-8">

      <div className="flex flex-col md:flex-row gap-6 mb-10 items-start bg-white/70 backdrop-blur-lg p-6 rounded-3xl border border-green-200 shadow-lg hover:shadow-xl transition-shadow duration-300">
        <div className="relative w-36 h-full md:w-40 md:h-full rounded-3xl overflow-hidden shadow-inner border border-green-300 flex items-center justify-center">
          {crop.image ? (
            <img
              src={crop.image}
              alt={crop.name}
              className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
            />
          ) : (
            <span className="text-xs text-gray-300">No Image</span>
          )}
        </div>

        <div className="flex flex-col gap-4 md:gap-3 flex-1">
          <h2 className="text-3xl md:text-4xl font-extrabold text-green-900 tracking-tight">
            {crop.name}
          </h2>

          <p className="text-green-800 font-medium text-sm md:text-base uppercase">
            {crop.type?.toUpperCase()} · {crop.season?.toUpperCase()} SEASON
          </p>

          <div className="flex items-start gap-4">
            <div className="w-1.5 h-full bg-green-500 rounded-full"></div>
            <p className="text-gray-700 text-sm md:text-base leading-relaxed max-w-xl">
              {overview}
            </p>
          </div>
        </div>
      </div>


      <div className="flex items-center gap-2 mb-4">
        <FiClipboard className="text-green-800 text-xl" />
        <h3 className="text-lg font-bold text-green-900">The Growth Journey</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {steps.map((step, i) => (
          <div
            key={i}
            className="relative bg-green-50 border border-green-200 rounded-xl p-4 shadow-sm"
          >
            <div className="absolute -top-3 -left-3 w-8 h-8 bg-green-600 text-white flex items-center justify-center rounded-full">
              {i + 1}
            </div>

            <p className="mt-4 text-sm font-semibold text-gray-700">
              {step}
            </p>
          </div>
        ))}

        <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 col-span-full text-center">
          <FiZap className="text-amber-600 text-xl mx-auto" />
          <p className="font-semibold text-amber-700 mt-2">Happy Harvesting!</p>
          <p className="text-sm text-amber-600">Enjoy your fresh crops</p>
        </div>
      </div>
    </div>
  );
}


