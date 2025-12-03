// ChatbotIcon.jsx
import { FaRobot } from "react-icons/fa";
import { NavLink } from "react-router-dom";
import { useSelector } from "react-redux";
import { useState } from "react";

export default function ChatbotIcon() {
  const { currentUser } = useSelector((state) => state.user);
  const [hovered, setHovered] = useState(false);

  const link = currentUser ? "/chatbot" : "/accesspage";

  return (
    <NavLink to={link}>
      <div
        className="fixed bottom-6 right-6 z-50 flex items-center space-x-4"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <div
          className={`overflow-hidden whitespace-nowrap bg-green-50 text-green-900 font-medium px-4 py-2 rounded-lg shadow-lg border-2 border-green-400 transition-all duration-300 ease-in-out ${
            hovered
              ? "max-w-xs opacity-100 translate-x-0"
              : "max-w-0 opacity-0 -translate-x-2"
          }`}
        >
          Chat with me!
        </div>

        <button
          className="bg-green-600 hover:bg-green-700 text-white p-4 rounded-full shadow-xl border-4 border-green-400 transition-all duration-300 ease-in-out transform hover:scale-110 hover:shadow-2xl relative"
        >
          <FaRobot size={34} />
          <span className="absolute top-0 left-0 w-full h-full rounded-full bg-green-400 opacity-30 animate-ping"></span>
        </button>
      </div>
    </NavLink>
  );
}
