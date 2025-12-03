import React, { useState } from "react";
import { NavLink, Link } from "react-router-dom";
import logo from "../assets/logo.png";
import * as LucideIcons from "lucide-react";
import { useSelector } from "react-redux";

export default function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const { currentUser } = useSelector((state) => state.user);

  const baseLink =
    "flex items-center text-gray-200 hover:text-yellow-300 transition-colors duration-200 px-3 py-2 text-sm font-medium rounded-lg";
  const activeLink = "text-yellow-300 font-bold bg-green-800/50";
  const desktopTextSize = "text-xs";
  const mobileTextSize = "text-sm";

  const getDesktopLinkClasses = ({ isActive }) =>
    isActive
      ? `${baseLink} ${activeLink} ${desktopTextSize}`
      : `${baseLink} ${desktopTextSize}`;

  const getMobileLinkClasses = ({ isActive }) =>
    isActive
      ? `${baseLink} ${activeLink} ${mobileTextSize}`
      : `${baseLink} ${mobileTextSize}`;

  const closeMenu = () => {
    setIsOpen(false);
  };


  const NavIcon = ({ icon: Icon }) => <Icon className="h-4 w-4 mr-1.5" />;

  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-green-900 shadow-xl border-b border-green-700/50">
      <div className="container mx-auto flex justify-between items-center px-4 sm:px-6 py-3">
        {/* Logo */}
        <div className="flex items-center space-x-3">
          <img
            src={logo}
            alt="SmartAgri Logo"
            className="h-8 w-8 rounded-full shadow-lg "
          />
          <NavLink to="/" onClick={closeMenu}>
            <span className="text-white font-extrabold text-base tracking-wider hover:text-yellow-300 transition-colors">
              SmartAgri
            </span>
          </NavLink>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center space-x-2 lg:space-x-4">
          <NavLink to="/" className={getDesktopLinkClasses}>
            <NavIcon icon={LucideIcons.LayoutDashboard} />
            Dashboard
          </NavLink>

          <NavLink to="/croplibrary" className={getDesktopLinkClasses}>
            <NavIcon icon={LucideIcons.BookOpen} />
            Crops Library
          </NavLink>

          <NavLink to="/cropriskcalculater" className={getDesktopLinkClasses}>
            <NavIcon icon={LucideIcons.BarChart3} />
            Risk Analyzer
          </NavLink>

          <NavLink to="/croprecomnder" className={getDesktopLinkClasses}>
            <NavIcon icon={LucideIcons.ThumbsUp} />
            Recommender
          </NavLink>

          <NavLink to="/weather" className={getDesktopLinkClasses}>
            <NavIcon icon={LucideIcons.CloudSun} />
            Weather
          </NavLink>

          <NavLink to="/disease" className={getDesktopLinkClasses}>
            <NavIcon icon={LucideIcons.Microscope} />
            Disease Predictor
          </NavLink>
        </nav>

        {/* User / Sign-in & Mobile Toggle */}
        <div className="flex items-center space-x-4">
          <div className="hidden md:block">
            {!currentUser ? (
              <Link
                to="/signin"
                className="bg-yellow-400 text-green-900 px-4 py-2 rounded-full font-bold text-sm shadow-md hover:bg-yellow-300 transition-colors flex items-center"
              >
                <LucideIcons.LogIn className="h-4 w-4 mr-2" />
                Get Started
              </Link>
            ) : (
              <Link to="/profile" className="flex items-center group">
                <img
                  src={currentUser.profilePicture || "/default-avatar.png"}
                  alt="Profile"
                  className="h-8 w-8 rounded-full border-2 border-yellow-400 object-cover shadow-lg group-hover:border-yellow-300 transition-colors"
                />
              </Link>
            )}
          </div>

          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-white focus:outline-none p-1 rounded-md hover:bg-green-700 transition-colors"
            aria-label="Toggle menu"
          >
            {isOpen ? <LucideIcons.X size={24} /> : <LucideIcons.Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <div
        className={`md:hidden absolute w-full bg-green-900/95 backdrop-blur-sm text-gray-200 border-t border-green-800 transition-all duration-300 ease-in-out ${isOpen ? "max-h-screen opacity-100 py-4" : "max-h-0 opacity-0 overflow-hidden"
          }`}
      >
        <div className="px-4 sm:px-6 space-y-2">
          <NavLink to="/" onClick={closeMenu} className={getMobileLinkClasses}>
            <NavIcon icon={LucideIcons.LayoutDashboard} />
            Dashboard
          </NavLink>

          <NavLink
            to="/croplibrary"
            onClick={closeMenu}
            className={getMobileLinkClasses}
          >
            <LucideIcons.BookOpen className="h-4 w-4 mr-2" />
            Crops Library
          </NavLink>



          <NavLink
            to="/cropriskcalculater"
            onClick={closeMenu}
            className={getMobileLinkClasses}
          >
            <NavIcon icon={LucideIcons.CloudRain} />
            Risk Analyzer
          </NavLink>


          <NavLink to="/croprecomnder" onClick={closeMenu} className={getMobileLinkClasses}>
            <NavIcon icon={LucideIcons.ThumbsUp} />
            Crop Recommender
          </NavLink>
          <NavLink to="/weather" onClick={closeMenu} className={getMobileLinkClasses}>
            <NavIcon icon={LucideIcons.CloudSun} />
            Weather Center
          </NavLink>
          <NavLink to="/disease" onClick={closeMenu} className={getMobileLinkClasses}>
            <NavIcon icon={LucideIcons.Microscope} />
            Disease Prediction
          </NavLink>

          {/* User/Sign-in (Mobile) */}
          <div className="pt-4 border-t border-green-800">
            {!currentUser ? (
              <Link
                to="/signin"
                onClick={closeMenu}
                className="block bg-yellow-400 text-green-900 text-center py-2 rounded-full font-bold shadow-md hover:bg-yellow-300 transition flex items-center justify-center text-sm"
              >
                <LucideIcons.LogIn className="h-4 w-4 mr-2" />
                Get Started
              </Link>
            ) : (
              <Link
                to="/profile"
                onClick={closeMenu}
                className="flex items-center space-x-3 p-2 bg-green-800/50 rounded-lg"
              >
                <img
                  src={currentUser.profilePicture || "/default-avatar.png"}
                  alt="Profile"
                  className="h-8 w-8 rounded-full border-2 border-yellow-400 object-cover"
                />
                <span className="text-white font-semibold">View Profile</span>
              </Link>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
