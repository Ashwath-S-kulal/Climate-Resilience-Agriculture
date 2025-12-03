// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: "mern-auth-e623e.firebaseapp.com",
  projectId: "mern-auth-e623e",
  storageBucket: "mern-auth-e623e.firebasestorage.app",
  messagingSenderId: "522397456920",
  appId: "1:522397456920:web:db6bafc15e57650549416d"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);