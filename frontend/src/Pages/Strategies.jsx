import React from 'react';


// Data structure containing all the climate resilience techniques
const resilienceData = [
    {
        category: "Water Management",
        icon: "💧",
        color: "bg-blue-600",
        detail: "Focuses on minimizing water loss, maximizing absorption, and efficient storage to survive droughts and manage excess rain.",
        items: [
            { title: "Contour Farming / Terracing", subIcon: "⛰️", description: "Involves plowing and planting parallel to the natural slope contours. Terraces are built as steps on steep slopes. This dramatically slows down water runoff, increasing the time for infiltration and significantly reducing soil erosion during heavy rains.", imageUrl: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRymRTuZ0reB_VjC_2GiYXsAsW0MUR83y6lYA&s" },
            { title: "Small Ponds / Farm Pits", subIcon: "🕳️", description: "Strategically located small reservoirs or pits dug on the farm to capture and store rainwater or surface runoff during the wet season. This stored water is crucial for supplementary irrigation during dry spells, acting as a buffer against drought.", imageUrl: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT9G3wWu1kEJFf-5oEN2TS_vmC4593kkaqLpw&s" },
            { title: "Alternate Wetting and Drying (AWD) for Rice", subIcon: "🌧️", description: "A water-saving technology where irrigation water is applied only when the water level drops to a certain depth below the soil surface, instead of keeping the field continuously flooded. This saves up to 30% of water, reduces irrigation costs, and significantly lowers methane (a potent greenhouse gas) emissions.", imageUrl: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSEQ_kK22pDuuGbj5s_-_T7ariOjXswUbdZxQ&s" },
        ]
    },
    {
        category: "Soil & Fertility Management",
        icon: "🌱",
        color: "bg-green-600",
        detail: "Strategies centered on building healthy soil structure, improving water retention, and naturally increasing nutrient content.",
        items: [
            { title: "Green Manures / Legume Intercropping", subIcon: "🌿", description: "Planting non-cash crops (like legumes or clover) specifically to be incorporated into the soil (green manure) or growing legumes alongside cash crops (intercropping). Legumes host nitrogen-fixing bacteria, naturally enriching the soil with nitrogen, reducing the need for chemical fertilizers, and improving overall soil structure.",  imageUrl: "#" },
            { title: "Biochar Addition", subIcon: "🔥", description: "Adding biochar (a charcoal-like material made from organic matter pyrolysis) to the soil. Biochar has a highly porous structure, which dramatically improves the soil's capacity to retain water and nutrients, making the soil more fertile and resilient to both drought and nutrient leaching.",  imageUrl: "#" },
            { title: "No-Till or Minimum Tillage", subIcon: "🚜", description: "Planting crops directly into undisturbed soil or with minimal soil disturbance. This leaves organic residue (previous crop stubble) on the surface, which acts as a protective mulch. It drastically reduces soil erosion from wind/rain and preserves soil moisture by limiting evaporation, especially vital during drought periods.",  imageUrl: "#"},
        ]
    },
    {
        category: "Crop Management",
        icon: "🌾",
        color: "bg-yellow-600",
        detail: "Techniques focused on optimizing planting cycles and crop diversity to maximize yield stability and adaptation to fluctuating climates.",
        items: [
            { title: "Relay Cropping", subIcon: "🔄", description: "The practice of planting a second crop into a standing first crop before the first crop is harvested. This maximizes the use of the growing season, increases total output from the same land, and ensures faster recovery or succession in case the first crop fails due to unseasonal weather.",  imageUrl: "#" },
            { title: "Intercropping", subIcon: "🤝", description: "Growing two or more different crops simultaneously on the same field (e.g., maize and beans). This biological diversity reduces the overall risk of complete crop loss, suppresses weeds, and improves soil health. Different root systems access water and nutrients at varying depths, increasing resource efficiency.",  imageUrl: "#" },
            { title: "Crop Rotation", subIcon: "🗓️", description: "Systematically changing the type of crops grown in a field over a sequence of seasons. This interrupts the life cycles of pests and diseases specific to one crop, prevents the depletion of specific soil nutrients, and helps maintain long-term soil fertility and structure.", imageUrl: "#" },
        ]
    },
    {
        category: "Pest & Disease Management",
        icon: "🐞",
        color: "bg-red-600",
        detail: "Using natural and integrated methods to manage threats that can be exacerbated by temperature and moisture changes.",
        items: [
            { title: "Trap Crops", subIcon: "🦟", description: "Planting a small area with a crop that is highly attractive to a specific pest to draw the pests away from the main, cash crop. Pests can then be physically destroyed or treated only on the small trap crop area, minimizing pesticide use on the main field.",  imageUrl: "#" },
            { title: "Neem or Garlic Sprays", subIcon: "🧄", description: "Using botanical extracts like neem oil (Azadirachta indica) or garlic as natural, biodegradable pesticides and fungicides. These are low-cost, effective against a wide range of pests, and their residues are safer for the environment and varying weather conditions compared to synthetic chemicals.",  imageUrl: "#" },
            { title: "Resilient Varieties", subIcon: "💪", description: "Selecting and cultivating crop varieties that have been bred or genetically adapted to be inherently resistant to common local pests, or diseases (like rusts or mildews) that thrive in specific weather-related conditions (e.g., prolonged wetness or heat stress).",  imageUrl: "#" },
        ]
    },
    {
        category: "Temperature & Extreme Weather",
        icon: "☀️",
        color: "bg-amber-600",
        detail: "Physical modifications to the farm environment to shield crops from harsh winds, intense heat, and untimely frost.",
        items: [
            { title: "Windbreaks", subIcon: "🌬️", description: "Rows of trees, shrubs, or artificial barriers planted strategically around fields. They reduce wind speed, preventing wind erosion, reducing desiccation (drying out) of soil and plants, and physically protecting sensitive crops from wind damage during storms.",  imageUrl: "#" },
            { title: "Shade Cloth / Agroforestry", subIcon: "🌳", description: "Using artificial netting (shade cloth) or integrating trees into farming systems (Agroforestry) to provide partial shade to high-value or heat-sensitive crops (like coffee or certain vegetables). This protects them from intense solar radiation and heat stress, maintaining growth and quality during heatwaves.",  imageUrl: "#" },
            { title: "Microclimate Farming", subIcon: "🌄", description: "Deliberately planting crops in specific areas of the farm where the local environment offers natural protection. For instance, planting heat-sensitive crops in low-lying, naturally shaded spots, or areas protected from the afternoon sun, to minimize temperature stress.",  imageUrl: "#" },
            { title: "Frost Protection", subIcon: "❄️", description: "Methods used to raise the temperature around the crop when frost is predicted. This can include running sprinklers (which release latent heat as the water freezes), or using temporary row covers (tunnels made of fabric or plastic) to trap ground heat, reducing damage to delicate plant tissue.",  imageUrl: "#" },
        ]
    },
    {
        category: "Low-Cost Innovative Ideas",
        icon: "💡",
        color: "bg-purple-600",
        detail: "Simple, practical, and inexpensive methods leveraging basic materials and traditional knowledge to enhance resilience.",
        items: [
            { title: "Bottle Drip Irrigation", subIcon: "🍼", description: "A DIY irrigation method where plastic bottles are cut, filled with water, and placed upside down next to plants with a small hole in the cap or base. This delivers water slowly and directly to the root zone, minimizing evaporation and maximizing water efficiency at a zero hardware cost.",  imageUrl: "#" },
            { title: "Sand Mulching", subIcon: "🏖️", description: "Applying a layer of coarse sand or small gravel on the soil surface, particularly effective in sandy or arid regions. This inert layer prevents the fine soil underneath from drying out by reducing capillary rise and evaporation, acting as a non-organic mulch to retain moisture.",  imageUrl: "#"},
            { title: "Planting in Mounds", subIcon: "🗻", description: "Creating raised beds or mounds of soil, especially beneficial in areas prone to seasonal flooding or high water tables. This keeps the plant roots and crown well above the standing water, preventing waterlogging and root rot, ensuring aeration and survival.",  imageUrl: "#" },
            { title: "Seed Priming", subIcon: "🧪", description: "Pre-soaking seeds in water (sometimes with nutrient solutions) for a controlled period and then surface-drying them before planting. This kick-starts the germination process, ensuring rapid and uniform emergence when planted, which is crucial for survival under potential drought or heat-stress conditions post-planting.", imageUrl: "#" },
        ]
    },
];

const App = () => {
    return (
        <div className="min-h-screen bg-blue-200 p-4 sm:p-8 font-sans">
            <header className="text-center mb-12">
                <h2 className="text-3xl font-bold text-gray-800 mb-12 text-center border-b-4 border-emerald-300 pb-4">
                    sustainable agricultural practices for adapting to extreme weather and climate change.
                </h2>
            </header>

            <main className="max-w-7xl mx-auto">

                {resilienceData.map((category, index) => (
                    <div className='bg-gray-300/15 rounded-lg shadow-2xl'>
                        <section key={index} className="mb-16 p-5">
                            {/* 1. Category Header */}
                            <div className={`flex items-center p-4 rounded-t-xl shadow-lg ${category.color} text-white`}>
                                <div className="text-3xl mr-4">{category.icon}</div>
                                <div>
                                    <h2 className="text-3xl font-bold">{category.category}</h2>
                                    <p className="text-sm opacity-90 italic mt-1">{category.detail}</p>
                                </div>
                            </div>




                            {/* 3. Technique Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-10">
                                {category.items.map((item, itemIndex) => (
                                    <div key={itemIndex} className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition duration-300 transform hover:-translate-y-1 border-t-4 border-gray-200 flex flex-col justify-between">

                                        <div>
                                            {/* Technique Title and Icon */}
                                            <div className="flex items-center mb-3">
                                                <span className={`text-4xl p-2 rounded-lg mr-4 ${category.color.replace('-600', '-100')} ${category.color.replace('bg-', 'text-')} flex-shrink-0`}>
                                                    {item.subIcon}
                                                </span>
                                                <h3 className={`text-xl font-semibold ${category.color.replace('bg-', 'text-')}`}>
                                                    {item.title}
                                                </h3>
                                            </div>

                                            {/* Full-Width Image Section for Individual Field */}
                                            <div className="w-full h-28 bg-gray-100 mb-4 rounded-lg overflow-hidden border border-gray-200 shadow-inner">
                                                <img
                                                    src={item.imageUrl}
                                                    alt={`Visual for ${item.title}`}
                                                    className="w-full h-full object-cover"
                                                    onError={(e) => { e.target.onerror = null; e.target.src = "https://placehold.co/400x150/f3f4f6/9ca3af?text=Image+Error" }}
                                                />
                                            </div>

                                            {/* Description */}
                                            <p className="text-gray-700 leading-relaxed text-sm mb-4">
                                                {item.description}
                                            </p>
                                        </div>

                                        {/* Know More Button */}
                                        <a
                                            href={`https://www.google.com/search?q=${encodeURIComponent(item.title + " climate resilience farming")}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className={`mt-auto inline-block text-center px-4 py-2 text-sm font-semibold rounded-lg shadow-md transition duration-150 ease-in-out text-white ${category.color} hover:opacity-90 hover:shadow-lg focus:outline-none focus:ring-4 focus:ring-offset-2 ${category.color.replace('-600', '-300')}`}
                                        >
                                            Know More →
                                        </a>

                                    </div>
                                ))}
                            </div>
                        </section> </div>
                ))}

            </main>


        </div>
    );
};

export default App;