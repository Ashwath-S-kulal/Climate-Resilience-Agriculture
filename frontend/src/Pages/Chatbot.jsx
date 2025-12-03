import React, { useState, useRef, useEffect } from "react";
import Header from "../Components/Header";
import { Bot, Brain, MessageSquare, SendHorizontal } from "lucide-react";
import { FaRobot } from "react-icons/fa";
import { useSelector } from "react-redux";

export default function Chatbot (){
  const [messages, setMessages] = useState([
    {
      sender: "assistant",
      text: "Hello! I'm your agricultural assistant. How can I help you today?",
      time: new Date().toLocaleTimeString(),
    },
  ]);

  const [input, setInput] = useState("");
  const { currentUser } = useSelector((state) => state.user);
  const [loading, setLoading] = useState(false);



  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    setLoading(true);
    if (input.trim() === "") return;

    const newMessage = {
      sender: "user",
      text: input,
      time: new Date().toLocaleTimeString(),
    };
    setMessages([...messages, newMessage]);
    setInput("");

    try {
      const response = await fetch("/api/ai/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        { sender: "assistant", text: data.reply, time: new Date().toLocaleTimeString() },
      ]);
    } catch (err) {
      console.error("Error fetching AI response:", err);
      setMessages((prev) => [
        ...prev,
        { sender: "assistant", text: "Error fetching response.", time: new Date().toLocaleTimeString() },
      ]);
    }
    setLoading(false);

  };

  return (
    <div className="flex flex-col min-h-screen w-full bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] text-gray-900">
      <Header />
      <div className="flex flex-col flex-1 mt-20 px-4 md:px-8">
        <div className="border-b border-white/40 pb-3 mb-2 flex items-center justify-center">
          <span className="h-3 w-3 rounded-full bg-green-500 mr-2 animate-pulse"></span>
          <span className="font-semibold text-black drop-shadow text-sm md:text-base">
            AI Farming Assistant — Online
          </span>
        </div>

        <div className="w-full flex justify-center">
          <div className="w-full max-w-[650px]">

            <div
              className="flex-1 overflow-y-auto space-y-5 mb-6 max-h-[70vh] px-2 scrollbar-thin scrollbar-thumb-green-600/60 scrollbar-track-transparent"
            >
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex items-end gap-2 ${msg.sender === "assistant" ? "flex-row" : "flex-row-reverse"
                    }`}
                >
                  {msg.sender === "assistant" ? (
                    <div
                      className="w-8 h-8 rounded-full border border-green-300 shadow 
            bg-white flex items-center justify-center"
                    >
                      <FaRobot size={18} className="text-green-700" />
                    </div>
                  ) : (
                    <img
                      src={currentUser.profilePicture || "/default-avatar.png"}
                      className="w-8 h-8 rounded-full border border-gray-300 shadow object-cover"
                    />
                  )}

                  {/* Chat Bubble */}
                  <div className="flex flex-col max-w-[75%]">
                    <div
                      className={`
                                  px-4 py-2.5 rounded-2xl shadow-md backdrop-blur-xl border text-sm break-words ${msg.sender === "assistant"
                          ? "bg-white/70 text-black rounded-bl-none border-green-200"
                          : "bg-green-700 text-white rounded-br-none border-green-300"
                        }`}
                    >
                      {msg.text}
                    </div>


                    <span
                      className={`text-[10px] mt-1 ${msg.sender === "assistant"
                        ? "text-black/50 ml-1"
                        : "text-black/50 mr-1 self-end"
                        }`}
                    >
                      {msg.time}
                    </span>
                  </div>
                </div>
              ))}

              {/* ✅ AI TYPING LOADER (ONLY WHEN loading === true) */}
              {loading && (
                <div className="flex items-end gap-2">
                  {/* Bot Avatar */}
                  <div
                    className="w-8 h-8 rounded-full border border-green-300 shadow 
          bg-white flex items-center justify-center"
                  >
                    <FaRobot size={18} className="text-green-700" />
                  </div>

                  {/* Typing Bubble */}
                  <div
                    className="px-4 py-2.5 rounded-2xl shadow-md backdrop-blur-xl border text-sm
          bg-white/70 text-black border-green-200 rounded-bl-none"
                  >
                    <div className="flex gap-1 items-center">
                      <span className="w-2 h-2 bg-green-600 rounded-full animate-bounce"></span>
                      <span className="w-2 h-2 bg-green-600 rounded-full animate-bounce delay-150"></span>
                      <span className="w-2 h-2 bg-green-600 rounded-full animate-bounce delay-300"></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>


          </div>
        </div>

        <div className="fixed bottom-4 left-0 w-full flex justify-center px-4 z-50">
          <div
            className="flex flex-row items-center max-w-[650px] w-full bg-white/40 backdrop-blur-xl border border-white/30 p-2 rounded-full shadow-lg">
            <input
              type="text"
              placeholder="Ask anything..."
              className="flex-1 px-4 py-2 bg-transparent outline-none text-black placeholder-black/60"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              disabled={loading} // optional: disable typing while loading
            />

            <button
              onClick={handleSend}
              disabled={loading}
              className={`rounded-full p-3 shadow-md transition active:scale-95  ${loading ? "bg-gray-400 cursor-not-allowed" : "bg-green-600 hover:bg-green-700 text-white"}`}
            >
              {loading ? (
                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
              ) : (
                <SendHorizontal size={20} />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>

  );
};


