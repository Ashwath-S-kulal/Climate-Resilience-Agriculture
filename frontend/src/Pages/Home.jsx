import React from 'react';
import WellcomeCard from "../Components/WellcomeCard"
import About from "./About";
import { NavLink } from 'react-router-dom';
import ChatbotIcon from '../Components/ChatbotIcon';

const colorMap = {
  "blue-500": {
    bg: "bg-blue-500", text: "text-blue-500", ring: "focus:ring-blue-300",
    shadow: "hover:shadow-blue-500/50", border: "border-blue-500"
  },
  "yellow-500": {
    bg: "bg-yellow-500", text: "text-yellow-500", ring: "focus:ring-yellow-300",
    shadow: "hover:shadow-yellow-500/50", border: "border-yellow-500"
  },
  "pink-500": {
    bg: "bg-pink-500", text: "text-pink-500", ring: "focus:ring-pink-300",
    shadow: "hover:shadow-pink-500/50", border: "border-pink-500"
  },
  "green-500": {
    bg: "bg-green-500", text: "text-green-500", ring: "focus:ring-green-300",
    shadow: "hover:shadow-green-500/50", border: "border-green-500"
  },
  "red-500": {
    bg: "bg-red-500", text: "text-red-500", ring: "focus:ring-red-300",
    shadow: "hover:shadow-red-500/50", border: "border-red-500"
  },
  "indigo-500": {
    bg: "bg-indigo-500", text: "text-indigo-500", ring: "focus:ring-indigo-300",
    shadow: "hover:shadow-indigo-500/50", border: "border-indigo-500"
  },
  "gray-500": {
    bg: "bg-gray-500", text: "text-gray-500", ring: "focus:ring-gray-300",
    shadow: "hover:shadow-gray-500/50", border: "border-gray-500"
  },
};


export default function Home() {
  const Leaf = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.4 3 15.6 15.1 14.6 17.5" /><path d="M17 19c-2.2 0-3.6-.8-4.4-1.6c-.6-.7-.9-1.5-.9-2.2c0-.8.1-1.6.4-2.4c.3-.8.8-1.5 1.4-2.2c.6-.7 1.3-1.4 2.2-2.1c.9-.6 1.8-1.2 2.7-1.7c.9-.5 1.9-.8 2.8-1c.9-.2 1.8-.3 2.7-.3c.9 0 1.8.1 2.7.3c.9.2 1.9.5 2.8 1c.9.5 1.8 1.1 2.7 1.7c.9.7 1.6 1.4 2.2 2.1c.6.7 1.1 1.4 1.4 2.2c.3.8.4 1.6.4 2.4c0 .7-.3 1.5-.9 2.2c-.8.8-2.2 1.6-4.4 1.6" /></svg>
  );
  const Info = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>
  );
  const AlertTriangle = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
  );
  const CalendarClock = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 7.5V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h3.5" /><path d="M16 2v4" /><path d="M8 2v4" /><path d="M3 10h5" /><path d="M18 22a6 6 0 1 0 0-12 6 6 0 0 0 0 12z" /><path d="M18 14v4l2 1" /></svg>
  );
  const Lightbulb = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 14c.2-1 .5-2 1.5-3c.8-.8 1.4-2 1.8-3" /><path d="M9 14c-.2-1-.5-2-1.5-3c-.8-.8-1.4-2-1.8-3" /><path d="M2 17c1.4.6 3.8 1 5.5 0" /><path d="M22 17c-1.4.6-3.8 1-5.5 0" /><path d="M12 21a2 2 0 0 1-2-2v-1h4v1a2 2 0 0 1-2 2z" /><path d="M12 18v-5c0-1.7 1.3-3 3-3s3 1.3 3 3v5" /><path d="M12 18v-5c0-1.7-1.3-3-3-3s-3 1.3-3 3v5" /></svg>
  );
  const Bug = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 20.8A7 7 0 1 0 10 3.2v17.6z" /><path d="M14 20.8A7 7 0 1 1 14 3.2v17.6z" /><line x1="12" y1="2" x2="12" y2="22" /><path d="M8 12h8" /><path d="M22 8l-3 3" /><path d="M19 13l3 3" /><path d="M2 8l3 3" /><path d="M5 13l-3 3" /></svg>
  );
  const CloudSun = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v2" /><path d="M12 20v2" /><path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" /><path d="M2 12h2" /><path d="M20 12h2" /><path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" /><path d="M15.5 13.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z" /><path d="M7 16a3.5 3.5 0 0 1 3.5-3.5h7A4.5 4.5 0 0 1 20 17c0 2.5-2 5-4.5 5H9c-3 0-5-1.5-5-4s1.5-3.5 3.5-3.5z" /></svg>
  );
  const CheckCircle = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><path d="m9 11 3 3L22 4" /></svg>
  );

  const features = [
    {
      icon: AlertTriangle,
      title: "Crop Risk Analyzer",
      description: "Real-time analysis based on current and forecast weather to identify and mitigate immediate risks to your crop health and investment.",
      category: "Planning",
      color: "blue-500",
      navLink: "/cropriskcalculater",
    },
    {
      icon: Lightbulb,
      title: "Crop Recommender",
      description: "Intelligent suggestions for the most suitable crops to plant now, maximizing resilience and profitability based on current patterns.",
      category: "Planning",
      color: "yellow-500",
      navLink: "/croprecomnder",
    },
    {
      icon: CalendarClock,
      title: "Crop Resilient Tips",
      description: "A deeply detailed guide , linking each principle directly to its foundational source and essential reading for sustainable farming.",
      category: "Planning",
      color: "pink-500",
      navLink: "/croplibrary",
    },
    {
      icon: Leaf,
      title: "Crop Steps to Grow",
      description: "Detailed, stage-by-stage cultivation instructions optimized for your region and specific environmental conditions and soil type.",
      category: "Planning",
      color: "green-500",
      navLink: "/croplist",
    },
    {
      icon: Bug,
      title: "Crop Disease Predictor",
      description: "Uses environmental and local data to predict the probability of common crop diseases, providing early warnings and preventive treatments.",
      category: "Defense",
      color: "red-500",
      navLink: "/disease",
    },
    {
      icon: CloudSun,
      title: "Hyper-Local Weather Forecast",
      description: "Accurate, minute-by-minute and 10-day forecasts tailored specifically to your farm's location to inform critical daily decisions.",
      category: "Defense",
      color: "indigo-500",
      navLink: "/weather",
    },
    {
      icon: Info,
      title: "Crop Information Provider",
      description: "A comprehensive knowledge base offering insights on crop genetics, nutritional needs, best harvest practices, and market viability.",
      category: "Knowledge",
      color: "gray-500",
      navLink: "/cropinfo",
    },

  ];

  const benefits = [
    "Maximize Yields with Data-Driven Insights",
    "Reduce Losses from Climate Volatility",
    "Optimize Resource Usage (Water, Fertilizer)",
    "Implement Proactive Disease and Pest Management",
    "Ensure Crop Selection Matches Future Climate Projections",
    "Simplify Complex Farm Planning and Scheduling",
  ];

  const FeatureCard = ({ icon: Icon, title, description, color, navLink }) => {
    const classes = colorMap[color] || colorMap["green-500"];
    console.log(Icon)

    return (
      <NavLink
        to={navLink}
        className={`flex flex-col text-left p-6 bg-white rounded-xl shadow-lg transition-all duration-300 transform hover:scale-[1.02] ${classes.shadow} group border border-gray-100 cursor-pointer focus:outline-none focus:ring-4 ${classes.ring} h-full`}
      >
        <div className="flex items-start mb-4">
          <div className={`p-3 rounded-lg ${classes.bg} text-white`}>
            <Icon />
          </div>
        </div>

        <h3 className="text-lg font-bold text-gray-800 mt-2">{title}</h3>
        <p className="mt-2 text-gray-600 text-sm leading-relaxed flex-grow">{description}</p>

        <div className="mt-4">
          <span className={`text-xs font-semibold ${classes.text} transition-colors flex items-center`}>
            Launch Tool &rarr;
          </span>
        </div>
      </NavLink>
    );
  };


  const planningFeatures = features.filter(f => f.category === "Planning");
  const defenseFeatures = features.filter(f => f.category === "Defense");
  const knowledgeFeature = features.find(f => f.category === "Knowledge");

  const knowledgeClasses = colorMap[knowledgeFeature?.color] || colorMap["gray-500"];
  const KnowledgeIcon = knowledgeFeature?.icon || Info;

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] font-sans">


      <header className="relative bg-gradient-to-br from-green-700 to-green-900 pt-10 pb-32 text-white shadow-2xl md:rounded-b-3xl">
        <section >
          <WellcomeCard />
        </section>
      </header>

      <main className="max-w-7xl mx-auto px-4 -mt-16 relative z-20">
        <section className="relative mb-20">
          <div className="absolute inset-0 bg-white rounded-3xl shadow-2xl transform skew-y-1 lg:skew-y-0 lg:skew-x-1 -z-10"></div>
          <div className="bg-white p-8 sm:p-12 rounded-3xl shadow-2xl relative z-10 grid grid-cols-1 lg:grid-cols-2 gap-10">

            <div>
              <h3 className="text-2xl font-bold text-gray-800 mb-4 border-b-2 border-green-500 pb-2">Our Mission</h3>
              <div className="space-y-4 text-base text-gray-600">
                <p>
                  To provide farmers, large and small, with unparalleled predictive intelligence necessary to combat the unpredictability of climate change. We aim to secure global food systems by making sustainable, high-yield farming accessible through technology.
                </p>
                <p className="text-sm italic">
                  We bridge the gap between complex climate science and practical agricultural execution, ensuring every decision on the farm is backed by reliable data.
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-2xl font-bold text-gray-800 mb-4 border-b-2 border-blue-500 pb-2">Our Vision</h3>
              <div className="space-y-4 text-base text-gray-600">
                <p>
                  We envision a future where climate change is no longer the primary threat to agriculture. By utilizing advanced AI and machine learning models, we will create fully autonomous, adaptive farming plans that optimize growth cycles and maximize resource efficiency in real-time.
                </p>
                <p className="text-sm italic">
                  Our goal is to be the global standard for resilient farming, helping ecosystems flourish alongside profitable businesses.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-20">
          <h2 className="text-3xl font-extrabold text-center text-gray-900 mb-4">
            Phase 1: Intelligent Planning
          </h2>
          <p className="text-center text-base text-gray-600 mb-12 max-w-3xl mx-auto">
            Optimize your growing season from the first seed to the final harvest with our smart planning suite.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {planningFeatures.map((feature, index) => (
              <FeatureCard key={index} {...feature} />
            ))}
          </div>
        </section>

        <section className="mb-20 bg-gray-100 p-8 sm:p-12 rounded-3xl shadow-inner">
          <h2 className="text-3xl font-extrabold text-center text-gray-900 mb-4">
            Phase 2: Defense & Prediction
          </h2>
          <p className="text-center text-base text-gray-600 mb-12 max-w-3xl mx-auto">
            Proactively mitigate risks by utilizing hyper-local climate and biological prediction models.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {defenseFeatures.map((feature, index) => (
              <FeatureCard key={index} {...feature} />
            ))}
          </div>
        </section>

        <section className="mb-20">
          <h2 className="text-3xl font-extrabold text-center text-gray-900 mb-4">
            Phase 3: The Knowledge Core
          </h2>
          <p className="text-center text-base text-gray-600 mb-12 max-w-3xl mx-auto">
            The foundation of smart farming—a centralized, verifiable repository of agricultural data.
          </p>
          <div className='flex gap-10'>
            <div className="max-w-4xl mx-auto mb-7">
              <NavLink
                to={knowledgeFeature.navLink}
                className={`flex flex-col md:flex-row items-center p-10 bg-white rounded-3xl shadow-2xl border-l-8 ${knowledgeClasses.border} focus:outline-none focus:ring-4 ${knowledgeClasses.ring}`}
              >
                <div className={`p-5 rounded-full ${knowledgeClasses.bg} text-white flex-shrink-0 mb-6 md:mb-0 md:mr-6`}>
                  <KnowledgeIcon />
                </div>
                <div className="text-center md:text-left mr-10">
                  <h3 className="text-xl font-bold text-gray-800">{knowledgeFeature.title}</h3>
                  <p className="mt-3 text-sm text-gray-600">{knowledgeFeature.description}</p>
                </div>
                <span className={`mt-4 md:mt-0 md:ml-auto text-sm font-bold ${knowledgeClasses.text} flex-shrink-0`}>
                  Access Data &rarr;
                </span>
              </NavLink>
            </div>

            <div className="max-w-4xl mx-auto">
              <NavLink
                to="/croplibrary"
                className={`flex flex-col md:flex-row items-center p-10 bg-white rounded-3xl shadow-2xl border-l-8 ${knowledgeClasses.border} focus:outline-none focus:ring-4 ${knowledgeClasses.ring}`}
              >
                <div className={`p-5 rounded-full ${knowledgeClasses.bg} text-white flex-shrink-0 mb-6 md:mb-0 md:mr-6`}>
                  <KnowledgeIcon />
                </div>
                <div className="text-center md:text-left mr-10">
                  <h3 className="text-xl font-bold text-gray-800">Resilience Tips and Tricks</h3>
                  <p className="mt-3 text-sm text-gray-600">A deeply detailed guide , linking each principle directly to its foundational source and essential reading for sustainable farming.</p>
                </div>
                <span className={`mt-4 md:mt-0 md:ml-auto text-sm font-bold ${knowledgeClasses.text} flex-shrink-0`}>
                  Access Data &rarr;
                </span>
              </NavLink>
            </div>
          </div>
        </section>

        <section className="bg-green-800 text-white p-10 sm:p-16 rounded-2xl shadow-2xl mb-20">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
            <div className="lg:col-span-1">
              <h3 className="text-3xl font-bold mb-4">Why AgriResilience?</h3>
              <p className="text-base opacity-90">
                We deliver precision agriculture with a focus on future climate adaptation. Our platform is built by climatologists and agricultural scientists.
              </p>
            </div>
            <div className="lg:col-span-2">
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
                {benefits.map((benefit, index) => (
                  <li key={index} className="flex items-start space-x-3 text-sm">
                    <CheckCircle className="flex-shrink-0 mt-1 w-6 h-6 text-yellow-400" />
                    <span>{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section className="text-center p-8 sm:p-12 bg-gray-100 rounded-2xl border border-gray-200 mb-10">
          <h2 className="text-3xl font-bold text-green-800 mb-4">
            Join the Movement. Farm Smarter.
          </h2>
          <p className="text-base text-gray-600 mb-8 max-w-3xl mx-auto">
            Tap into the power of climate science and machine learning to guarantee crop security and profitability in an uncertain world.
          </p>
          <NavLink to={"/croplibrary"}>
            <button
              className="px-10 py-4 bg-green-600 text-white text-xl font-bold rounded-full shadow-lg hover:bg-green-700 transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-green-300"
              onClick={() => console.log("Final Call to Action clicked")}
            >
              Explore More Details
            </button>
          </NavLink>
        </section>
      </main>
      <section>
        <About />
      </section>
    </div>
  );
};