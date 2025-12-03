// import React, { useEffect, useState } from "react";
// import Papa from "papaparse";

// export default function Fertilizer() {
//   const [data, setData] = useState([]);

//   useEffect(() => {
//     // Path to CSV file inside "public" folder
//     const csvPath = "../../public/data/All-India_-Crop-wise-Area,-Production-&-Yield.csv";

//     fetch(csvPath)
//       .then((response) => response.text())
//       .then((csvText) => {
//         Papa.parse(csvText, {
//           header: true,
//           skipEmptyLines: true,
//           complete: (results) => {
//             setData(results.data);
//           },
//         });
//       })
//       .catch((error) => console.error("Error loading CSV:", error));
//   }, []);

//   return (
//     <div className="p-6 bg-gray-50 min-h-screen">
//       <h1 className="text-2xl font-bold mb-4">Crop Data (Auto Loaded)</h1>

//       {data.length > 0 ? (
//         <div className="overflow-x-auto">
//           <table className="min-w-full border-collapse bg-white shadow-md rounded-lg">
//             <thead className="bg-blue-600 text-white">
//               <tr>
//                 {Object.keys(data[0]).map((col, idx) => (
//                   <th key={idx} className="px-4 py-2 border">
//                     {col}
//                   </th>
//                 ))}
//               </tr>
//             </thead>
//             <tbody>
//               {data.map((row, i) => (
//                 <tr key={i} className="hover:bg-gray-100">
//                   {Object.values(row).map((value, j) => (
//                     <td key={j} className="px-4 py-2 border text-sm">
//                       {value}
//                     </td>
//                   ))}
//                 </tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       ) : (
//         <p className="text-gray-600">Loading CSV data...</p>
//       )}
//     </div>
//   );
// }
