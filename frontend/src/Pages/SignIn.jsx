import React, { useState } from "react";
import { FaApple, FaAppleAlt } from "react-icons/fa";
import Header from "../Components/Header";
import { useNavigate } from "react-router-dom";
import {
  signInStart,
  signInSuccess,
  signInFailure,
} from "../redux/user/userSlice";
import { useDispatch, useSelector } from "react-redux";
import OAuth from "../Components/OAuth";
import { FaGoogle } from "react-icons/fa";


export default function SigninForm() {
  const [formData, setFormData] = useState({});
  const { loading, error } = useSelector((state) => state.user);

  const navigate = useNavigate();
  const dispatch = useDispatch();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      dispatch(signInStart());
      const res = await fetch("/api/auth/signin", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      const data = await res.json();

      if (data.success === false) {
        dispatch(signInFailure(data));
        return;
      }

      dispatch(signInSuccess(data));
      navigate("/");
    } catch (error) {
      dispatch(signInFailure(error));
      alert("Invalid Credential. Please try again.");
    }
  };

  return (
     <div className="min-h-screen bg-gradient-to-br from-green-950 via-emerald-900 to-green-800 flex items-center justify-center px-4 pt-14">
      <Header/>
      
      <div className="mt-5 w-full max-w-5xl bg-gray-800/60 rounded-2xl overflow-hidden shadow-2xl flex flex-col md:flex-row border border-white/10 backdrop-blur-xl">

        {/* ✅ LEFT — Image section */}
        <div className="relative w-full md:w-1/2">
          <img
            src="https://eng.kisanofindia.com/wp-content/uploads/2022/12/Copy-of-Untitled-78.jpg"
            alt="Farm Landscape"
            className="w-full h-full object-cover brightness-90"
          />

          {/* Overlay */}
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 to-black/70"></div>

          <div className="absolute bottom-6 left-6 text-white">
            <h1 className="text-3xl font-bold">
              Capturing Nature,
              <br /> Growing the Future
            </h1>
            <p className="mt-2 text-sm text-green-200/80">
              Sustain your land. Sustain your future.
            </p>
          </div>
        </div>

        {/* ✅ RIGHT — Form section */}
        <div className="w-full md:w-1/2 p-10 text-white">
          <h2 className="text-3xl font-bold text-white">
            Welcome Back
          </h2>

          <p className="text-sm text-gray-300 mt-2">
            Login to continue
          </p>

          {/* FORM */}
          <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
            <div>
              <label className="block text-sm font-medium text-gray-300">
                Email
              </label>
              <input
                type="email"
                id="email"
                placeholder="Enter your email"
                onChange={handleChange}
                required
                className="w-full p-2.5 mt-1 bg-gray-700/40 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300">
                Password
              </label>
              <input
                type="password"
                id="password"
                placeholder="Enter your password..."
                onChange={handleChange}
                required
                className="w-full p-2.5 mt-1 bg-gray-700/40 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:outline-none"
              />
            </div>

            <button
              disabled={loading}
              type="submit"
              className="w-full bg-green-600 hover:bg-green-700 text-white py-2.5 rounded-md font-semibold text-sm transition shadow-md hover:shadow-lg"
            >
              {loading ? "Signing In..." : "Sign In"}
            </button>
          </form>

          {/* Divider */}
          <div className="flex items-center my-6">
            <div className="flex-grow h-px bg-gray-600" />
            <span className="px-3 text-xs text-gray-400">OR</span>
            <div className="flex-grow h-px bg-gray-600" />
          </div>

          {/* OAuth Buttons */}
          <div className="flex gap-3">
            <button className="w-full  flex items-center justify-center gap-2 border border-gray-500 rounded-lg hover:bg-gray-700 transition">
              <OAuth/>
            </button>
        
          </div>

          <p className="mt-6 text-sm text-gray-300 text-center">
            Don’t have an account?{" "}
            <a
              href="/signup"
              className="text-green-400 hover:text-green-300 underline"
            >
              Sign Up
            </a>
          </p>

          <p className="text-red-500 mt-3 text-center text-xs">
            {error ? "Invalid Credentials.!" : ""}
          </p>
        </div>
      </div>
    </div>

  );
}
