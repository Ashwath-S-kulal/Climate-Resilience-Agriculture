import { Play, CheckCircle } from "lucide-react";

export default function AboutSection() {
  return (
    <section className="relative py-20 pb-10 px-6 lg:px-20 bg-gradient-to-r from-green-900 via-green-800 to-green-600 text-white overflow-hidden md:rounded-t-3xl">
      <div className="relative grid md:grid-cols-2 gap-12 items-center z-10">
        <div className="relative rounded-2xl shadow-2xl overflow-hidden group">
          <img
            src="https://media.istockphoto.com/id/1469639791/photo/farmer-using-digital-tablet-in-corn-crop-cultivated-field-with-smart-farming-interface-icons.jpg?s=612x612&w=0&k=20&c=CEnLHATfACNoH_N3Ru5KoTOmAc5SbufJozvV_kcuwc4="
            alt="Farmer with crops"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
          />

          <div className="absolute inset-0 bg-gradient-to-t from-green-900/70 via-green-800/40 to-transparent"></div>

          <div className="absolute inset-0 flex items-center justify-center">
            <button className="w-20 h-20 rounded-full bg-gradient-to-r from-green-400 to-yellow-400 flex items-center justify-center shadow-lg hover:scale-110 transition duration-300 border-4 border-white/40">
              <Play className="text-white w-8 h-8" />
            </button>
          </div>
        </div>

        <div className="space-y-6">

          <h2 className="text-2xl lg:text-3xl font-extrabold leading-snug drop-shadow-md">
            How We Care & Grow Our Organic Vegetables
          </h2>

          <p className="text-green-100/90 leading-relaxed">
            We’re committed to sustainable farming that nurtures both people and the planet.
            Discover how we combine traditional agricultural wisdom with modern climate-smart technologies
            to ensure healthy, thriving crops.
          </p>

          <ul className="space-y-4 text-green-100">
            <li className="flex items-start gap-3">
              <CheckCircle className="text-green-300 w-6 h-6 flex-shrink-0" />
              <p>Climate-smart monitoring for soil and weather to optimize yield naturally.</p>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle className="text-green-300 w-6 h-6 flex-shrink-0" />
              <p>Organic fertilizer use ensuring sustainability and soil health.</p>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle className="text-green-300 w-6 h-6 flex-shrink-0" />
              <p>Empowering local farmers with AI-driven precision agriculture insights.</p>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}
