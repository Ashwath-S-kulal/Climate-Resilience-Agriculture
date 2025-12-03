import React from "react";
import BGfooter from "../assets/bgfooter.webp";
import logo from "../assets/logo.png";
import { Phone, Mail, MapPin, Facebook, Twitter, Instagram } from "lucide-react";

const teamMembers = [
  {
    name: "Amshu H U",
    phone: "+91 70221 58133",
    email: "amshugowda177@gmail.com",
  },
  {
    name: "Apoorva",
    phone: "+91 6361 672 543",
    email: "apoorvabhagwath@gmail.com",
  },
  {
    name: "Ashwath S",
    phone: "+91 8431294514",
    email: "ashwathkulal2004@gmail.com",
  },
  {
    name: "Manu Sagar",
    phone: "+91 95352 00895",
    email: "manusagar030@gmail.com",
  },
];

export default function Footer() {
  return (
    <footer
      className="relative text-white"
      style={{
        backgroundImage: `url(${BGfooter})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      <div className="bg-black/70">
        <div className="max-w-7xl mx-auto px-6 py-5 md:py-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          <div>
            <div className="flex items-center mb-4">
              <img
                src={logo}
                alt="AgriClimate"
                className="w-12 h-12 mr-3 rounded-xl"
              />
              <h2 className="text-2xl font-bold">SmartAgri</h2>
            </div>
            <p className="text-gray-300 text-sm">
              Sustainable and climate-smart agriculture solutions to help
              farmers grow more efficiently while caring for the planet.
            </p>
            <div className="flex space-x-3 mt-4">
              <a
                href="#"
                className="p-2 rounded-full bg-green-600 hover:bg-green-500 transition"
              >
                <Facebook className="w-4 h-4 text-white" />
              </a>
              <a
                href="#"
                className="p-2 rounded-full bg-green-600 hover:bg-green-500 transition"
              >
                <Twitter className="w-4 h-4 text-white" />
              </a>
              <a
                href="#"
                className="p-2 rounded-full bg-green-600 hover:bg-green-500 transition"
              >
                <Instagram className="w-4 h-4 text-white" />
              </a>
              <span className="bg-green-500 p-2 rounded-lg">
                <MapPin className="w-4 h-4 text-white" />
              </span>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4 relative">
              Credits
              <span className="block w-12 h-1 bg-green-500 mt-1 rounded"></span>
            </h3>
            <div className="md:col-span-2 lg:col-span-3 flex flex-col md:flex-row gap-8">

              {teamMembers.map((member, idx) => (
                <div key={idx}>
                  <h3 className="text-lg font-medium mb-2 relative">
                    {member.name}

                  </h3>
                  <ul className="space-y-2 text-gray-300 text-sm">
                    <li className="flex items-center gap-2">
                      <Phone className="w-4 h-4 text-white" />
                      <span>{member.phone}</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <Mail className="w-4 h-4 text-white" />
                      <span>{member.email}</span>
                    </li>
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="bg-gray-900 py-4 text-center text-sm text-gray-400">
          Copyright © 2025 AgriClimate. All rights reserved.
        </div>
      </div>
    </footer>
  );
}
