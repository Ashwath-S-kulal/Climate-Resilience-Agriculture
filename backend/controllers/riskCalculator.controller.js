import axios from "axios";
import Crop from "../models/cropSensitivity.model.js";
import {
  calculateRisk,
  generateRecommendations,
} from "../utils/riskCalculator.js";

export const riskCalculater = async (req, res) => {
  const { place, crop } = req.body;

  if (!place || !crop) {
    return res.status(400).json({ error: "place and crop are required" });
  }

  try {
    // Fetch crop sensitivity data from MongoDB
    const cropSensitivity = await Crop.find({}).lean();
    console.log("Crop Sensitivity Data");

    const geoRes = await axios.get(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
        place
      )}&format=json&limit=1`,
      { headers: { "User-Agent": "Climate-App" } }
    );

    if (!geoRes.data || geoRes.data.length === 0) {
      return res.status(404).json({ error: "Location not found" });
    }

    const { lat, lon, display_name } = geoRes.data[0];
    const latitude = parseFloat(lat);
    const longitude = parseFloat(lon);

    const weatherRes = await axios.get(
      `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&daily=temperature_2m_max,precipitation_sum&timezone=auto`
    );

    const weatherData = {
      temperature: weatherRes.data.daily.temperature_2m_max[0],
      rainfall: weatherRes.data.daily.precipitation_sum[0],
    };

    const riskScores = calculateRisk(
      weatherData,
      crop.toLowerCase(),
      cropSensitivity
    );

    const recommendations = generateRecommendations(
      riskScores,
      crop.toLowerCase()
    );

    res.json({
      location: { place: display_name, latitude, longitude },
      weatherData,
      riskScores,
      recommendations,
    });
  } catch (err) {
    console.error("Error in riskCalculater:", err);
    res.status(500).json({ error: "Something went wrong!" });
  }
};
