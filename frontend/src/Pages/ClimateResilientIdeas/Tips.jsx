import React, { useState } from "react";
import { Leaf, Droplet, CloudRain, Users, Check } from "lucide-react";
import Header from "../../Components/Header";

export default function Tips() {
  const tips = [
    {
      id: 'soil',
      title: "Soil Health & Ecology",
      subtitle: "The foundational layer for nutrient cycling and carbon sequestration.",
      icon: Leaf,
      color: "text-teal-500",
      tips: [
        "Implement no-till or reduced tillage practices to preserve soil structure and microbial life, minimizing carbon release.",
        "Use diverse cover cropping systems (legumes, grasses) to enhance nitrogen fixation and prevent wind/water erosion.",
        "Manage residue effectively to build soil organic matter and significantly increase water infiltration capacity.",
        "Regularly test soil to accurately target nutrient deficiencies, maximizing fertilizer efficiency and avoiding chemical excess.",
        "Incorporate biochar or local compost to rapidly boost water retention and long-term carbon storage in the field."
      ]
    },
    {
      id: 'water',
      title: "Water Use & Efficiency",
      subtitle: "Strategic management of water resources for drought mitigation and supply security.",
      icon: Droplet, 
      color: "text-blue-500",
      tips: [
        "Transition to high-efficiency irrigation (sub-surface drip, microsprinklers) to minimize evaporation loss.",
        "Employ remote sensing and soil moisture probes for real-time, precision irrigation scheduling, providing water exactly when needed.",
        "Design and construct small-scale water harvesting structures (check dams, contour trenches) for surface runoff capture.",
        "Use thick organic mulches (straw, crop residues) to significantly reduce soil surface water evaporation and weed growth.",
        "Select crop varieties with naturally lower water requirements or enhanced drought tolerance suitable for local conditions."
      ]
    },
    {
      id: 'climate',
      title: "Climate & Weather Readiness",
      subtitle: "Adapting farming systems to handle weather extremes and increasing variability.",
      icon: CloudRain,
      color: "text-orange-500",
      tips: [
        "Diversify the farm system by integrating trees (agroforestry) and livestock, providing microclimates and multiple products.",
        "Utilize seasonal climate forecasts and early warning systems to make proactive decisions on planting and input allocation.",
        "Shift planting and harvesting dates to avoid predicted periods of extreme heat, frost, or intense rainfall events.",
        "Invest in protective measures like shade nets or high tunnels for vulnerable, high-value crops against hail and strong sun.",
        "Develop contingency plans for post-disaster recovery, including access to emergency seeds and equipment."
      ]
    },
    {
      id: 'economic',
      title: "Economic & Social Resilience",
      subtitle: "Future-proof your operation and build strong community and reliable market ties.",
      icon: Users,
      color: "text-indigo-500",
      tips: [
        "Diversify income streams through value-added processing, direct consumer market sales, or eco-tourism initiatives.",
        "Establish farmer cooperatives for bulk purchasing of expensive inputs and collective market negotiation power.",
        "Secure comprehensive crop insurance or risk-sharing mechanisms to buffer against weather-related income loss.",
        "Foster knowledge sharing, mentorship, and peer-to-peer training on new climate-smart agricultural techniques.",
        "Use robust financial planning to maintain an operational buffer fund and manage exposure to price volatility."
      ]
    }
  ];

  const [active, setActive] = useState("soil");
  const section = tips.find((s) => s.id === active);
  const ActiveIcon = section.icon;

  return (
    <div className="w-full bg-gray-50 ">
      <Header />
      <header className="w-full text-center pb-7 pt-24 bg-gradient-to-t from-green-800 to-green-600 text-white shadow-2xl ">
        <div className="max-w-6xl mx-auto px-4">
          <h1 className="text-4xl sm:text-4xl font-bold text-white mb-0 tracking-tight">
            Climate Resilience Framework
          </h1>

        </div>
      </header>


      <main className="max-w-6xl mx-auto pt-10">

        <div className="flex flex-wrap justify-between items-center space-x-2 md:space-x-4 mb-8 p-1 bg-white rounded-xl shadow-lg border border-gray-200">
          {tips.map((sec) => {
            const Icon = sec.icon;
            const isActive = sec.id === active;

            return (
              <button
                key={sec.id}
                onClick={() => setActive(sec.id)}
                className={`flex flex-col md:flex-row items-center flex-1 p-3 md:p-4 font-semibold rounded-lg transition ${isActive ? `bg-gray-100 ${sec.color}` : "text-gray-600 hover:bg-gray-50"
                  }`}
              >
                <Icon className={`w-5 h-5 md:mr-2 ${isActive ? sec.color : "text-gray-400"}`} />
                <span>{sec.title.split(" ")[0]}</span>
              </button>
            );
          })}
        </div>
        <div className="p-8 bg-white rounded-lg shadow-inner border fadeIn">
          <div className="flex items-center mb-6">
            <ActiveIcon className={`w-8 h-8 mr-4 ${section.color}`} />
            <div>
              <h3 className="text-2xl font-bold">{section.title}</h3>
              <p className="text-gray-500">{section.subtitle}</p>
            </div>
          </div>

          <ul className="space-y-6">
            {section.tips.map((tip, i) => (
              <li key={i} className="flex">
                <span className={`${section.color} mr-4 mt-1`}>
                  <Check className="w-4 h-4" />
                </span>
                <p className="text-lg text-gray-700">
                  <strong>{i + 1}. </strong>{tip}
                </p>
              </li>
            ))}
          </ul>
        </div>
      </main>
    </div>
  );
};

