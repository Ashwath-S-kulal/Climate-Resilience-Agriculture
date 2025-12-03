import express from "express";
import mongoose from "mongoose";
import dotenv from "dotenv";
import userRoutes from "./routes/user.route.js";
import authRoutes from "./routes/auth.route.js";
import chatbotRoutes from "./routes/chatbot.route.js";
import cookieParser from "cookie-parser";
import predictionRoutes from "./routes/prediction.route.js";
import riskcalculateRoutes from "./routes/riskCalculator.route.js"
import cropReccomender from "./routes/cropRecommender.route.js"

dotenv.config();
const app = express();

app.use(express.json());
app.use(cookieParser()); 


app.use("/api/user", userRoutes);
app.use("/api/auth", authRoutes);
app.use("/api/ai", chatbotRoutes);
app.use("/api/predictions", predictionRoutes);
app.use("/api/calculate", riskcalculateRoutes);
app.use("/api/reccomender", cropReccomender);



const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("Server listening on port", PORT);
});


mongoose
  .connect(process.env.MONGO)
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error(err));
