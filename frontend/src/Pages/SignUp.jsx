import React from "react";
import Header from "../Components/Header";
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import OAuth from '../Components/OAuth';


export default function SignupForm() {
  const [formData, setFormData] = useState({});
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(false);
    console.log(formData);
    const res = await fetch('/api/auth/signup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData),
    });
    const data = await res.json();
    console.log(data);
    setLoading(false);

    if (data.success === false) {
      setError(true);
      return;
    }
    navigate('/signin');
  };
  return (

    <div className="min-h-screen bg-gradient-to-br from-green-950 via-emerald-900 to-green-800 flex items-center justify-center px-4 pt-14">
      <Header />

      <div className="w-full max-w-5xl bg-gray-800/60 rounded-2xl overflow-hidden shadow-xl flex flex-col md:flex-row border border-white/10 h-[80vh] backdrop-blur-xl">

        <div className="w-full md:w-1/2 px-9 py-8 text-white flex flex-col justify-center">
          <h2 className="text-2xl font-bold mb-1">Create Account</h2>
          <p className="text-xs text-gray-300 mb-6">Sign up to continue</p>

          <form className="space-y-3" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-medium">Name</label>
              <input
                id="username"
                type="text"
                placeholder="Enter your name"
                onChange={handleChange}
                className="w-full p-2.5 mt-1 bg-gray-700/40 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-medium">Email</label>
              <input
                id="email"
                type="email"
                placeholder="Enter your email"
                onChange={handleChange}
                className="w-full p-2.5 mt-1 bg-gray-700/40 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:outline-none" />
            </div>

            <div>
              <label className="block text-xs font-medium">Password</label>
              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                onChange={handleChange}
                className="w-full p-2.5 mt-1 bg-gray-700/40 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:outline-none" />
            </div>

            <button
              disabled={loading}
              type="submit"
              className="w-full bg-green-600 hover:bg-green-700 text-white py-2 rounded-md font-semibold text-sm transition-all"
            >
              {loading ? "Loading..." : "Sign Up"}
            </button>
          </form>

          <div className="flex items-center my-4">
            <div className="flex-grow h-px bg-gray-600" />
            <span className="px-3 text-xs text-gray-300">OR</span>
            <div className="flex-grow h-px bg-gray-600" />
          </div>

          <div className="flex justify-center">
            <OAuth />
          </div>

          <p className="mt-4 text-xs text-gray-300 text-center">
            Already have an account?{" "}
            <a href="/signin" className="text-green-400 font-medium hover:text-green-300">
              Sign In
            </a>
          </p>

          <p className="text-red-500 mt-2 text-center text-xs">
            {error && "Something went wrong!"}
          </p>
        </div>

        <div className="relative w-full md:w-1/2 h-full">
          <img
            src="https://eng.kisanofindia.com/wp-content/uploads/2022/12/Copy-of-Untitled-78.jpg"
            alt="Farmer"
            className="w-full h-full object-cover"
          />

          <div className="absolute inset-0 bg-black/40"></div>

          <div className="absolute bottom-6 left-6 text-white">
            <h1 className="text-3xl font-bold leading-tight drop-shadow-md">
              Cultivating Growth,
              <br /> One Step at a Time
            </h1>
            <p className="mt-2 text-green-300/90 text-xs">
              Join our farming community and start your journey.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
