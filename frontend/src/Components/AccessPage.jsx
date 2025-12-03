import Header from "./Header";
import Image from "../assets/123.png";
import { NavLink } from "react-router-dom";

export default function AccessPage() {
  return (
    <section className="relative w-full min-h-screen bg-gradient-to-b from-green-900 to-green-950 text-white overflow-hidden flex items-center pt-12">
      <Header />
      <div className="absolute w-72 h-72 bg-gradient-to-br from-green-500 to-green-700 rounded-full blur-3xl opacity-30 left-[-80px] bottom-10 animate-pulse" />
      <div className="absolute w-52 h-52 bg-gradient-to-tr from-green-400 to-lime-400 rounded-full blur-2xl opacity-20 right-20 top-24 animate-pulse" />
      <div className="absolute w-40 h-40 bg-gradient-to-t from-green-600 to-lime-600 rounded-full blur-2xl opacity-20 right-1/3 bottom-0 animate-pulse" />
      <div className="absolute w-32 h-32 bg-gradient-to-t from-pink-500 to-red-400 rounded-full blur-xl opacity-20 left-1/3 top-10 animate-pulse" />

      <div className="max-w-7xl mx-auto px-6 py-16 grid grid-cols-1 md:grid-cols-2 gap-12 items-center leading-loose">
        <div className="text-center md:text-left">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight">
            Boost
            <br />
            <span className="text-green-400">Your Agriculture Business</span>
          </h1>
          <p className="mt-6 text-gray-300 max-w-md mx-auto md:mx-0 leading-7">
            Empower your farms with modern solutions, including Crop Management, Crop Recommender, Disease Prediction, and Weather Center, to increase crop yield, monitor weather, and streamline agricultural operations for a sustainable future.
          </p>

          <p className="mt-10 text-lg font-semibold text-white drop-shadow-md">
            Login to access these feature
          </p>

          <div className="mt-6 flex flex-col sm:flex-row gap-6 justify-center md:justify-start">
            <NavLink to={"/signin"}>
              <button className="px-6 py-3 rounded-xl text-lg bg-green-500 hover:bg-green-600 text-white transition">
                Sign In
              </button>
            </NavLink>
            <NavLink to={"/signup"}>
              <button className="px-6 py-3 rounded-xl text-lg border border-green-400 text-green-400 hover:bg-green-400 hover:text-gray-900 transition">
                Sign Up
              </button>
            </NavLink>
          </div>
        </div>

        <div
          className="flex justify-center relative w-full h-80 sm:h-96 lg:h-[500px] bg-cover bg-center"
          style={{ backgroundImage: `url(${Image})` }}
        />
      </div>
    </section>
  );
}
