import mongoose from "mongoose";

const predictionSchema = new mongoose.Schema({
  userId: {
    type: String,  
    required: true,
  },
  prediction: {
    type: String,
    required: true,
  },
  confidence: {
    type: Number,
    required: true,
  },
  date: {
    type: Date,
    default: Date.now,
  },
});

const Prediction = mongoose.model("Prediction", predictionSchema);

export default Prediction;
