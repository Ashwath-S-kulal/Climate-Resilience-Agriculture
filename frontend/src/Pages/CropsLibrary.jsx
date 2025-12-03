import React from "react";
import {
    Leaf,
    BarChart2,
    AlertTriangle,
    HelpCircle,
    Lightbulb,
    Clock,
    Users,
    Globe2,
    DollarSign,
    BookOpen,
    ArrowRight,
    User,
} from "lucide-react";
import Header from "../Components/Header";
import { NavLink } from "react-router-dom";

// const recommendedBooks = [
//     { title: "The Permaculture Handbook", author: "J. Russell", sourceUrl: "https://www.nazboo.com/wp-content/uploads/2017/05/ebook-The-Permaculture-Handbook.-Garden-Farming-for-Town-and-Country-by-Peter-Bane.pdf", coverPlaceholder: "https://m.media-amazon.com/images/I/71IEGjlIjWL._UF1000,1000_QL80_.jpg" },
//     { title: "No-Till Farming Systems", author: "D. Johnson", sourceUrl: "http://www.waswac.org.cn/waswac/rootfiles/2017/08/17/1499910848045231-1499910848048172.pdf", coverPlaceholder: "https://0.academia-photos.com/attachment_thumbnails/91931327/mini_magick20221003-1-1bdeodq.png?1664824800" },
//     { title: "Water-Wise Agriculture", author: "S. Patel", sourceUrl: "https://www.amazon.in/Water-Management-Conservation-Harvesting-Artificial/dp/8122422241", coverPlaceholder: "https://m.media-amazon.com/images/I/71RGBngukpL._SY385_.jpg" },
//     { title: "Carbon Sequestration", author: "E. Chen", sourceUrl: "https://scholar.google.co.in/scholar?q=Carbon+Sequestration+By+E.+Chen+pdf&hl=en&as_sdt=0&as_vis=1&oi=scholart", coverPlaceholder: "https://onlinelibrary.wiley.com/cms/asset/ce5d81ba-0020-4ad3-9b2e-7c668813f739/gcb.2009.15.issue-10.cover.gif" },
//     { title: "The Resilient Farm", author: "B. White", sourceUrl: "https://toc.library.ethz.ch/objects/pdf03/e05_978-1-60358-444-9_01.pdf", coverPlaceholder: "https://m.media-amazon.com/images/I/91n89mVbtTL._UF1000,1000_QL80_.jpg" },
//     { title: "Integrated Pest Mgmt", author: "K. Lee", sourceUrl: "https://www.researchgate.net/publication/255730153_Integrated_Pest_Management", coverPlaceholder: "https://0.academia-photos.com/attachment_thumbnails/38689899/mini_magick20190224-6867-1hevjtp.png?1551023838" },
//     { title: "Climate Change Adaptation", author: "M. Garcia", sourceUrl: "https://www.researchgate.net/publication/200032572_Adaptation_to_Climate_Change_in_the_Developing_World", coverPlaceholder: "https://images.routledge.com/common/jackets/crclarge/978113805/9781138054509.jpg" },
//     { title: "Soil Biology", author: "A. Singh", sourceUrl: "https://www.amazon.in/Advances-Applied-Bioremediation-Soil-Biology/dp/3642269176", coverPlaceholder: "https://media.springernature.com/w138/springer-static/cover/series/5138.jpg" },
//     { title: "Agroforestry Systems Design", author: "L. Taylor", sourceUrl: "https://www.researchgate.net/publication/380373985_Designing_Agroforestry_Systems_Chapter-10", coverPlaceholder: "https://m.media-amazon.com/images/I/714rJsfksTL.jpg" },
//     { title: "Regenerative Grazing", author: "C. Miller", sourceUrl: "https://www.foodandlandusecoalition.org/wp-content/uploads/2019/09/Regenerative-Agriculture-final.pdf", coverPlaceholder: "https://m.media-amazon.com/images/I/81n8JP7-OBL._UF1000,1000_QL80_.jpg" }
// ];

// const successfulFarmers = [
//     {
//         title: " pioneering work in chemical-free, traditional farming methods",
//         farmer: "Hukumchand Patidar	",
//         location: "Rajasthan, India",
//         description: "Successful organic farmer exporting his produce to multiple countries. Practices organic farming on a large scale.",
//         image: "https://cms.patrika.com/wp-content/uploads/2025/08/Hukumchand-Patidar.jpg",
//         fullStoryUrl: "https://en.wikipedia.org/wiki/Hukumchand_Patidar#:~:text=Hukumchand%20Patidar%20(born%20c.,village%20in%20Jhalawar%20district%2C%20Rajasthan."
//     },
//     {
//         title: "transforming rural livelihoods through innovative agriculture ",
//         farmer: "Rajkumari Devi (Kisan Chachi)",
//         location: " Bihar, India",
//         description: "Mobilized over 300 women into Self-Help Groups (SHGs) for kitchen farming and value-added products (jams, pickles). Recipient of the Padma Shri award.",
//         image: "https://images.bhaskarassets.com/web2images/521/2024/02/04/whatsapp-image-2024-02-04-at-42900-pm_1707057967.jpg",
//         fullStoryUrl: "https://en.wikipedia.org/wiki/Kisan_Chachi"
//     },
//     {
//         title: "Introduced advanced and more profitable farming techniques",
//         farmer: "Ram Saran Verma ",
//         location: "Uttar Pradesh, India",
//         description: "He has been called the rural high-tech farmer by the Indian media, for introducing advanced farming techniques to improve yields in banana, tomato and potato crops for small rural Indian farmers.",
//         image: "https://sugermint.com/wp-content/uploads/2022/06/Ram-Saran-Varma.jpg",
//         fullStoryUrl: "https://en.wikipedia.org/wiki/Ram_Saran_Verma"
//     },
//     {
//         title: " Zero Budget Natural Farming (ZBNF)",
//         farmer: "Subhash Palekar	",
//         location: "Maharashtra, India",
//         description: "Developed SPNF, a method that eliminates chemical fertilizers and pesticides and instead relies on indigenous resources and traditional practices.",
//         image: "https://cms.thewire.in/wp-content/uploads/2017/07/subhash-palekar.jpg",
//         fullStoryUrl: "https://en.wikipedia.org/wiki/Subhash_Palekar"
//     },
// ];


export default function CropLibrary() {
    const iconBaseClass = "w-10 h-10 rounded-full flex items-center justify-center mb-3 transition duration-300 ease-in-out";
    const cardBaseClass = "bg-[#F0F0F0] rounded-xl shadow-2xl p-5 transition duration-300 ease-in-out transform hover:scale-[1.01] hover:shadow-lg cursor-pointer";
    const buttonBaseClass = "text-white px-3 py-1.5 rounded-lg w-full font-medium flex items-center justify-center transition duration-300 ease-in-out";

    return (
        <div className="w-full min-h-screen bg-gradient-to-b from-[#d8f3dc] via-[#b7e4c7] to-[#95d5b2] text-gray-800 font-sans">
            <Header />
            <div className="bg-gradient-to-t from-green-800 to-green-600 text-white py-12 pb-8 pt-24 ">
                <div className="max-w-7xl mx-auto px-6 text-center">
                    <h1 className="text-4xl font-extrabold mb-3 tracking-tight">
                        Climate Resilience Agriculture <span className="text-yellow-300">Knowledge Core</span>
                    </h1>
                </div>
            </div>
            <div className="bg-slate-200">
                <section className="max-w-full py-8 sm:py-12 pb-4 px-4 sm:px-6 md:mx-auto lg:mx-10">
                    <h2 className="text-xl sm:text-2xl font-extrabold mb-5 text-gray-900 ">
                        Core Management Tools
                    </h2>

                    <div className="grid grid-cols  -1 md:grid-cols-3 gap-6 sm:gap-7">
                        <NavLink to={"/croplist"}>
                            <div className={`${cardBaseClass} border-t-8 border-green-600 shadow-xl sm:shadow-2xl`}>
                                <div className={`${iconBaseClass} bg-green-100`}>
                                    <Leaf className="w-5 h-5 text-green-700" />
                                </div>
                                <h3 className="text-lg sm:text-xl font-bold mb-1 text-gray-900">Crops Growing Steps</h3>
                                <p className="text-xs sm:text-sm text-gray-600 mb-4">
                                    The Crop Steps to Grow Center provides complete, step-by-step guidance for cultivating any crop.
                                </p>
                                <button className={`${buttonBaseClass} bg-green-600 hover:bg-green-700`}>
                                    Explore Guide <ArrowRight className="ml-2 w-4 h-4" />
                                </button>
                            </div>
                        </NavLink>

                        <NavLink to={"/cropinfo"}>
                            <div className={`${cardBaseClass} border-t-8 border-green-600 shadow-xl sm:shadow-2xl`}>
                                <div className={`${iconBaseClass} bg-green-100`}>
                                    <BarChart2 className="w-5 h-5 text-green-700" />
                                </div>
                                <h3 className="text-lg sm:text-xl font-bold mb-1 text-gray-900">Crop Information Center</h3>
                                <p className="text-xs sm:text-sm text-gray-600 mb-4">
                                    The Crop Information Center helps farmers and users easily access detailed information about various crops.
                                </p>
                                <button className={`${buttonBaseClass} bg-green-600 hover:bg-green-700`}>
                                    Explore Data <ArrowRight className="ml-2 w-4 h-4" />
                                </button>
                            </div>
                        </NavLink>

                        <NavLink to={"/diseasedata"}>
                            <div className={`${cardBaseClass} border-t-8 border-green-600 shadow-xl sm:shadow-2xl`}>
                                <div className={`${iconBaseClass} bg-green-100`}>
                                    <AlertTriangle className="w-5 h-5 text-green-700" />
                                </div>
                                <h3 className="text-lg sm:text-xl font-bold mb-1 text-gray-900">Disease Data Dashboard</h3>
                                <p className="text-xs sm:text-sm text-gray-600 mb-4">
                                    This tool is designed to offer fast guidance and support so users can make informed decisions about their health.
                                </p>
                                <button className={`${buttonBaseClass} bg-green-600 hover:bg-green-700`}>
                                    Monitor Alerts <ArrowRight className="ml-2 w-4 h-4" />
                                </button>
                            </div>
                        </NavLink>
                    </div>
                </section>

                <hr className="max-w-screen mx-auto border-gray-300" />
                <section className="max-w-full px-4 sm:px-6 py-8 sm:py-12 md:mx-auto lg:mx-10">
                    <h2 className="text-xl sm:text-2xl font-extrabold mb-5 text-gray-900">
                        Business & Sustainability Hub
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-10 lg:gap-16">
                        <NavLink to={"/tips"}>
                            <div className={`${cardBaseClass} border-t-8 border-blue-500 shadow-xl sm:shadow-2xl`}>
                                <div className={`${iconBaseClass} bg-blue-100`}>
                                    <HelpCircle className="w-5 h-5 text-blue-600" />
                                </div>
                                <h3 className="text-lg sm:text-xl font-bold mb-1 text-gray-900">Resilient Tips & Tricks</h3>
                                <p className="text-xs sm:text-sm text-gray-600 mb-4">
                                    Simple tools and smart techniques can greatly improve your resilience and efficiency.
                                </p>
                                <button className={`${buttonBaseClass} bg-blue-600 hover:bg-blue-700`}>
                                    Get Tips <ArrowRight className="ml-2 w-4 h-4" />
                                </button>
                            </div>
                        </NavLink>

                        <NavLink to={"/stratergies"}>
                            <div className={`${cardBaseClass} border-t-8 border-blue-600 shadow-xl sm:shadow-2xl`}>
                                <div className={`${iconBaseClass} bg-blue-100`}>
                                    <Lightbulb className="w-5 h-5 text-blue-700" />
                                </div>
                                <h3 className="text-lg sm:text-xl font-bold mb-1 text-gray-900">Sustainable Practices</h3>
                                <p className="text-xs sm:text-sm text-gray-600 mb-4">
                                    Sustainable farming practices help protect the environment while improving crop productivity.
                                </p>
                                <button className={`${buttonBaseClass} bg-blue-600 hover:bg-blue-700`}>
                                    View Methods <ArrowRight className="ml-2 w-4 h-4" />
                                </button>
                            </div>
                        </NavLink>

                        <NavLink to={"/adaptation"}>
                            <div className={`${cardBaseClass} border-t-8 border-blue-600 shadow-xl sm:shadow-2xl`}>
                                <div className={`${iconBaseClass} bg-blue-100`}>
                                    <Clock className="w-5 h-5 text-blue-700" />
                                </div>
                                <h3 className="text-lg sm:text-xl font-bold mb-1 text-gray-900">Adaptation Strategies</h3>
                                <p className="text-xs sm:text-sm text-gray-600 mb-4">
                                    Adaptation strategies help farmers adjust to changing climate conditions and reduce risks.
                                </p>
                                <button className={`${buttonBaseClass} bg-blue-600 hover:bg-blue-700`}>
                                    Start Planning <ArrowRight className="ml-2 w-4 h-4" />
                                </button>
                            </div>
                        </NavLink>
                    </div>
                </section>
            </div>

            {/* <hr className="max-w-screen mx-auto border-gray-300" />

            <section className="max-w-7xl mx-auto py-10 sm:py-16 px-4 sm:px-6">
                <h2 className="text-2xl sm:text-3xl font-extrabold mb-6 sm:mb-10 text-gray-900 border-b-2 border-indigo-500 pb-2">
                    Recommended Reading
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-6 sm:gap-10 lg:gap-14">
                    {recommendedBooks.map((book, idx) => (
                        <div
                            key={idx}
                            className={`${cardBaseClass} shadow-md p-4 sm:p-5 rounded-xl border-t-4 border-indigo-600 flex flex-col items-center text-center bg-white hover:shadow-lg transition-all duration-200 h-full`}
                        >
                            <div className="w-24 h-32 sm:w-28 sm:h-36 mb-4 border bg-gray-100 rounded-md overflow-hidden">
                                <img src={book.coverPlaceholder} alt={book.title} className="object-cover w-full h-full" />
                            </div>

                            <p className="text-sm font-bold mb-1 text-gray-900">{book.title}</p>
                            <p className="text-xs text-gray-500 italic mb-4">By {book.author}</p>

                            <div className="mt-auto w-full">
                                <a
                                    href={book.sourceUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-2 rounded-md w-full text-xs font-semibold shadow block"
                                >
                                    View Source →
                                </a>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            <hr className="max-w-screen mx-auto border-gray-300" />

            <section className="max-w-full px-4 sm:px-6 lg:px-10 xl:px-36 py-10 sm:py-12 bg-slate-200">
                <h2 className="text-2xl sm:text-3xl font-extrabold mb-6 sm:mb-8 text-gray-900 border-b-2 border-orange-500 pb-2">
                    Farmer Success Stories
                </h2>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
                    {successfulFarmers.map((story, idx) => (
                        <div key={idx} className={`${cardBaseClass} border-l-4 border-orange-500 flex flex-col p-4`}>
                            <img
                                src={story.image}
                                alt={story.farmer}
                                className="w-full h-32 sm:h-36 object-cover rounded-md mb-3"
                            />

                            <div className="flex items-center mb-3">
                                <div className="w-8 h-8 rounded-full bg-orange-200 flex items-center justify-center text-orange-800 flex-shrink-0">
                                    <User className="w-4 h-4" />
                                </div>
                                <div className="ml-2 overflow-hidden">
                                    <p className="font-bold text-sm text-gray-900 truncate">{story.farmer}</p>
                                    <p className="text-xs text-gray-500 truncate">{story.location}</p>
                                </div>
                            </div>
                            <p className="text-sm font-semibold text-gray-900 mb-1">
                                {story.title}
                            </p>

                            <p className="text-sm italic mb-3 text-gray-700 line-clamp-3">
                                "{story.description}"
                            </p>

                            <a
                                href={story.fullStoryUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="bg-gray-200 hover:bg-gray-300 px-3 py-1.5 rounded-md w-full text-xs font-semibold text-center block mt-auto"
                            >
                                Read Full Story →
                            </a>

                        </div>
                    ))}
                </div>
            </section> */}
        </div>
    );
}
