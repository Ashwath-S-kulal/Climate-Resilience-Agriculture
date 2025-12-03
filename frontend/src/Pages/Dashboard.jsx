import React from "react";
import Contact from "./Home";
import Footer from "../Components/Footer";
import Header from "../Components/Header";
import ChatbotIcon from "../Components/ChatbotIcon";


export default function Dashboard() {

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-green-900 via-green-800 to-green-600 text-white pt-10"> 
      <Header />
      <div className="flex flex-1">
        <div className="flex-1 flex flex-col overflow-hidden">
          <section>
            <Contact />
          </section>
          <footer>
            <Footer />
          </footer>

        </div>
      </div>
      <ChatbotIcon/>
    </div>
  );
}
