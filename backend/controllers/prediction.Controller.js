import Prediction from "../models/Prediction.model.js";
import FormData from "form-data";
import axios from "axios";


export const predictDisease =async (req, res) => {
  try {
    console.log("Received file:", req.file);
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const formData = new FormData();
    formData.append("image", req.file.buffer, {
      filename: "image.jpg",
      contentType: req.file.mimetype
    });
    const response = await axios.post("http://localhost:5001/predict", formData, {
      headers: formData.getHeaders(),
    });

    res.json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
};

export const createPrediction = async (req, res) => {
  try {
    const { id } = req.params;
    const { prediction, confidence } = req.body;

    if (!id) {
      return res.status(400).json({ error: "ID parameter is required" });
    }

    if (!prediction || confidence === undefined) {
      return res.status(400).json({ error: "Prediction and confidence required" });
    }

    const newPrediction = new Prediction({
      userId: id,    
      prediction,
      confidence
    });

    await newPrediction.save();

    res.status(201).json({
      message: "Prediction saved successfully",
      data: newPrediction
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


export const getPredictions = async (req, res) => {
  try {
    const { id } = req.params;

    if (!id) {
      return res.status(400).json({ error: "ID parameter is required" });
    }

    const predictions = await Prediction.find({ userId: id }).sort({ date: -1 });

    res.json(predictions);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


export const deletePrediction = async (req, res) => {
  try {
    const { id } = req.params;
    const deleted = await Prediction.findByIdAndDelete(id);
    if (!deleted) {
      return res.status(404).json({ error: "Prediction not found" });
    }
    res.json({ message: "Prediction deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

export const clearPredictionHistory = async (req, res) => {
  try {
    const { id } = req.params;

    if (!id) {
      return res.status(400).json({ error: "User ID is required" });
    }

    // Delete all predictions for this user
    const deleted = await Prediction.deleteMany({ userId: id });

    res.status(200).json({
      message: "Prediction history cleared successfully",
      deletedCount: deleted.deletedCount
    });

  } catch (err) {
    console.error("Clear history error:", err);
    res.status(500).json({ error: err.message });
  }
};
