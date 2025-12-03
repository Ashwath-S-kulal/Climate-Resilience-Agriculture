// server/controllers/crop.controller.js (type: module)
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import csv from 'csv-parser';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const cropDataPath = path.join(__dirname, '../data/crops.csv');


export const getCropCatalog = async (req, res) => {
    try {
        const crops = [];
        
        fs.createReadStream(cropDataPath)
            .pipe(csv({
                mapValues: ({ header, value }) => {
                    if (['tMin', 'tMax', 'pMin', 'pMax'].includes(header)) {
                        return parseInt(value, 10);
                    }
                    return value;
                }
            }))
            .on('data', (data) => crops.push(data))
            .on('end', () => {
                res.status(200).json(crops);
            })
            .on('error', (err) => {
                console.error("CSV read error:", err);
                res.status(500).json({ message: "Failed to read crop catalog data." });
            });
    } catch (error) {
        console.error("Server error:", error);
        res.status(500).json({ message: "Internal server error." });
    }
};