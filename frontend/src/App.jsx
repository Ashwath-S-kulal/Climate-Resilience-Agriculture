import { BrowserRouter, Routes, Route } from 'react-router-dom';
import "./App.css"
import Dashboard from "./Pages/Dashboard";
import Home from "./Pages/Home"
import Weather from './Pages/Weather'
import Disease from './Pages/Disease';
import Chatbot from './Pages/Chatbot';
import About from './Pages/About';
import SignIn from './Pages/SignIn';
import SignUp from './Pages/SignUp';
import Profile from './Pages/Profile';
import PrivateRoute from './Components/PrivateRoute';
import CropsList from './Pages/CropList';
import AccessPage from './Components/AccessPage';
import CropRecomnder from './Pages/CropRecomnder';
import CropInfo from './Pages/CropInfo';
import CropRiskCalculater from './Pages/CropRiskCalculater';
import Tips from "./Pages/ClimateResilientIdeas/Tips"
import ScrollToTop from './Components/ScrollToTop';
import DiseaseData from "./Pages/DiseaseData"
import DiseaseSuppliment from "./Pages/DiseaseSuppliment"
import CropsLibrary from "./Pages/CropsLibrary"
import Stratergies from "./Pages/ClimateResilientIdeas/Practices"
import Adaptation from "./Pages/ClimateResilientIdeas/Adaptation"


export default function App() {
  return (
    <div>
      <BrowserRouter>
        <ScrollToTop />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/home" element={<Home />} />
          <Route path='/about' element={<About />} />
          <Route path='/signin' element={<SignIn />} />
          <Route path='/signup' element={<SignUp />} />
          <Route path="/croplist" element={<CropsList />} />
          <Route path='/accesspage' element={<AccessPage />} />
          <Route path='/cropinfo' element={<CropInfo />} />
          <Route path='/diseasedata' element={<DiseaseData />} />
          <Route path='/diseasesuppliment' element={<DiseaseSuppliment />} />
          <Route path='/croplibrary' element={<CropsLibrary />} />
          <Route path='/stratergies' element={<Stratergies />} />
          <Route path='/adaptation' element={<Adaptation />} />
          <Route path='/tips' element={<Tips />} />



          <Route element={<PrivateRoute />}>
            <Route path="/profile" element={<Profile />} />
            <Route path='/weather' element={<Weather />} />
            <Route path='/disease' element={<Disease />} />
            <Route path='/chatbot' element={<Chatbot />} />
            <Route path='/croprecomnder' element={<CropRecomnder />} />
            <Route path='/cropriskcalculater' element={<CropRiskCalculater />} />
          </Route>

          
        </Routes>
      </BrowserRouter>
    </div>
  )
}