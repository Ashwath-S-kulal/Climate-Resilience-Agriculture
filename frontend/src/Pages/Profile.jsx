import React from "react";
import { FaUserCircle } from "react-icons/fa";
import Header from "../Components/Header";
import { useSelector } from 'react-redux';
import { useState } from 'react';
import { useDispatch } from 'react-redux';
import {
  updateUserStart,
  updateUserSuccess,
  updateUserFailure,
  deleteUserStart,
  deleteUserSuccess,
  deleteUserFailure,
  signOut,
} from '../redux/user/userSlice';
import { FaUser, FaEnvelope, FaLock, FaSignOutAlt, FaTrash } from "react-icons/fa";
import { MdEdit } from "react-icons/md";


export default function ProfilePage() {
  const [formData, setFormData] = useState({});
  const [updateSuccess, setUpdateSuccess] = useState(false);
  const { currentUser, loading, error } = useSelector(state => state.user);
  const dispatch = useDispatch();



  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      dispatch(updateUserStart());
      const res = await fetch(`/api/user/update/${currentUser._id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (data.success === false) {
        dispatch(updateUserFailure(data));
        return;
      }
      dispatch(updateUserSuccess(data));
      setUpdateSuccess(true);
    } catch (error) {
      dispatch(updateUserFailure(error));
    }
  };

  const handleDeleteAccount = async () => {
    try {
      dispatch(deleteUserStart());
      const res = await fetch(`/api/user/delete/${currentUser._id}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (data.success === false) {
        dispatch(deleteUserFailure(data));
        return;
      }
      dispatch(deleteUserSuccess(data));
    } catch (error) {
      dispatch(deleteUserFailure(error));
    }
  };

  const handleSignOut = async () => {
    try {
      await fetch('/api/auth/signout');
      dispatch(signOut())
    } catch (error) {
      console.log(error);
    }
  };
  return (
    <div className="relative min-h-screen w-full bg-gradient-to-br from-[#001510] via-[#013d2f] to-[#015c49] overflow-hidden text-white">

      <Header />

      {/* AURORA BACKDROP */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-[60%] h-[40vh] bg-green-300/20 blur-3xl rounded-full animate-pulse"></div>
        <div className="absolute bottom-10 right-1/3 w-[50%] h-[40vh] bg-teal-400/20 blur-3xl rounded-full animate-pulse delay-300"></div>
        <div className="absolute top-1/3 right-0 w-[40%] h-[40vh] bg-yellow-400/20 blur-3xl rounded-full animate-pulse delay-700"></div>
      </div>

      {/* CONTENT WRAPPER */}
      <div className="relative w-full max-w-6xl mx-auto pt-20 pb-14 px-6">

        {/* TOP ACTION ROW */}
        <div className="flex items-center justify-between mb-10">
          <h1 className="text-4xl font-extrabold tracking-wide bg-gradient-to-r from-green-300 to-yellow-200 text-transparent bg-clip-text">
            Account Center
          </h1>

          <div className="flex gap-3">
            <button
              onClick={handleSignOut}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 border border-white/20 rounded-full hover:bg-white/20 transition text-sm"
            >
              <FaSignOutAlt /> Logout
            </button>

            <button
              onClick={handleDeleteAccount}
              className="flex items-center gap-2 px-4 py-2 bg-red-500/20 border border-red-400/30 text-red-200 rounded-full hover:bg-red-500/40 transition text-sm"
            >
              <FaTrash /> Delete
            </button>
          </div>
        </div>

        {/* MAIN TWO-PANEL LAYOUT */}
        <div className="w-full flex flex-col lg:flex-row gap-8">

          {/* LEFT — PROFILE CARD */}
          <div className="relative lg:w-1/3 bg-white/10 backdrop-blur-xl rounded-2xl border border-white/20 shadow-xl p-8 flex flex-col items-center">

            <div className="relative">
              {/* glowing aura */}
              <div className="absolute inset-0 bg-green-400/40 blur-2xl rounded-full animate-pulse"></div>

              <div className="relative w-40 h-40 rounded-full overflow-hidden ring-4 ring-green-300/50 shadow-2xl">
                <img
                  src={currentUser.profilePicture}
                  alt="Profile"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>

            <h2 className="text-2xl font-bold mt-5 flex gap-2 items-center">
              <FaUser /> {currentUser.username}
            </h2>

            <p className="mt-1 text-green-200/70 flex gap-2 items-center">
              <FaEnvelope /> {currentUser.email}
            </p>

            {/* Under-photo buttons */}
            <div className="absolute bottom-4 left-4">
              <button
                onClick={handleSignOut}
                className="p-3 bg-green-500/40 hover:bg-green-500/60 rounded-xl shadow-md"
              >
                <FaSignOutAlt className="text-xl" />
              </button>
            </div>

            <div className="absolute bottom-4 right-4">
              <button
                onClick={handleDeleteAccount}
                className="p-3 bg-red-500/40 hover:bg-red-500/60 rounded-xl shadow-md"
              >
                <FaTrash className="text-xl" />
              </button>
            </div>

          </div>


          {/* RIGHT — FORM PANEL */}
          <div className="lg:w-2/3 bg-white/5 backdrop-blur-xl rounded-2xl border border-white/20 shadow-xl p-10">

            <div className="flex items-center gap-3 mb-8">
              <MdEdit className="text-green-300 text-3xl" />
              <h2 className="text-3xl font-bold">Edit Profile</h2>
            </div>

            <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-8">

              {/* NAME */}
              <div className="relative">
                <FaUser className="absolute top-1/2 -translate-y-1/2 left-4 text-green-300" />
                <input
                  id="username"
                  defaultValue={currentUser.username}
                  onChange={handleChange}
                  className="w-full bg-[#00251c]/60 border border-green-400/20 rounded-xl px-12 py-4 text-white focus:ring-2 focus:ring-green-300 focus:border-green-300 placeholder-white/40 transition"
                  placeholder="Full Name"
                />
              </div>

              {/* EMAIL */}
              <div className="relative">
                <FaEnvelope className="absolute top-1/2 -translate-y-1/2 left-4 text-green-300" />
                <input
                  id="email"
                  defaultValue={currentUser.email}
                  onChange={handleChange}
                  className="w-full bg-[#00251c]/60 border border-green-400/20 rounded-xl px-12 py-4 text-white focus:ring-2 focus:ring-green-300 placeholder-white/40 transition"
                  placeholder="Email"
                />
              </div>

              {/* PASSWORD */}
              <div className="relative">
                <FaLock className="absolute top-1/2 -translate-y-1/2 left-4 text-green-300" />
                <input
                  id="password"
                  type="password"
                  defaultValue={currentUser.password}
                  onChange={handleChange}
                  className="w-full bg-[#00251c]/60 border border-green-400/20 rounded-xl px-12 py-4 text-white focus:ring-2 focus:ring-green-300 placeholder-white/40 transition"
                  placeholder="New Password"
                />
              </div>


              <button
                type="submit"
                className="mt-4 w-full rounded-xl bg-gradient-to-r from-green-400 to-green-600 py-4 font-bold shadow-lg hover:shadow-green-500/50 hover:scale-[1.03] transition-all"
              >
                {loading ? "Updating..." : "Save Changes"}
              </button>
            </form>

            {/* Status */}
            {error && <p className="text-red-400 mt-4 text-center">Something went wrong!</p>}
            {updateSuccess && <p className="text-green-400 mt-4 text-center">✅ Updated successfully!</p>}
          </div>
        </div>
      </div>
    </div>
    



  );
}