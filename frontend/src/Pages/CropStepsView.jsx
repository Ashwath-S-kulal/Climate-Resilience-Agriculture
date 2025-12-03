import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import { FiClipboard, FiZap } from "react-icons/fi";

function CropStepsView({ crop }) {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Papa.parse("../../public/data/Crop_Steps_Data.csv", {
      download: true,
      header: true,
      complete: (result) => {
        const rows = result.data;
        const found = rows.find((row) => row.id === String(crop.id));

        if (found) {
          setPlan({
            overview: found.overview,
            steps: found.steps?.split("|") ?? [],
          });
        } else {
          setPlan(null);
        }
        setLoading(false);
      },
    });
  }, [crop.id]);

  if (loading) return <p className="text-center mt-10 text-gray-600">Loading...</p>;

  if (!plan)
    return (
      <p className="text-white italic text-center mt-10 p-2 rounded-lg bg-green-700/50 text-sm">
        No growing data available for *{crop.name}* yet.
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

      <div className="flex items-center gap-2 mt-4 mb-4">
        <FiClipboard className="text-green-800 text-xl" />
        <h3 className="text-lg font-bold text-green-900">The Growth Journey</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 mt-4">
        {steps.map((step, i) => (
          <div
            key={i}
            className="relative bg-green-50 border border-green-200 rounded-xl p-4 shadow-sm"
          >
            <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center text-sm font-bold shadow">
              {i + 1}
            </div>

            <p className="text-sm font-semibold text-gray-700 mt-4 leading-relaxed">
              {step}
            </p>
          </div>
        ))}

        <div className="relative bg-amber-50 border border-amber-300 rounded-xl p-4 shadow-sm col-span-full">
          <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-amber-500 text-white flex items-center justify-center shadow">
            <FiZap className="text-sm" />
          </div>

          <p className="text-center font-semibold text-amber-700 mt-3">
             Happy Harvesting!
          </p>

          <p className="text-center text-sm text-amber-600 mt-1">
            Enjoy the fruits or vegetables! of your labor. Time to replant!
          </p>
        </div>
      </div>
    </div>
  );
}

export default CropStepsView;
