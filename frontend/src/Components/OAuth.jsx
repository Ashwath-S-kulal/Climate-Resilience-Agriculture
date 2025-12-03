import {getAuth, GoogleAuthProvider, signInWithPopup} from 'firebase/auth';
import {app} from '../firebase';
import {useDispatch} from 'react-redux';
import {signInSuccess} from '../redux/user/userSlice';
import {useNavigate} from 'react-router-dom';
import {FaGoogle } from 'react-icons/fa';
import logo from "../assets/GoogleImg.jpg"

export default function OAuth() {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const handleGoogleClick=async()=>{
        try{
            const  provider = new GoogleAuthProvider();
            const auth=getAuth(app);
            const result = await signInWithPopup(auth,provider);
            const res = await fetch('/api/auth/google',{
                method:"POST",
                headers:{
                    "Content-Type":"application/json",
                },
                body:JSON.stringify({
                    name:result.user.displayName,
                    email:result.user.email,
                    photo:result.user.photoURL
                }),
        });
        const data=await res.json();
        console.log(data);
        console.log(res);
        dispatch(signInSuccess(data));
        navigate('/');


        }catch(error){
            console.log("Cloud not login with google",error);
        }
    }
  return (
     <button className="bg-red-700 flex text-white items-center justify-center w-full border border-red-300 py-2 rounded-md hover:bg-red-800 " onClick={handleGoogleClick}>
                <img
src={logo}
  alt="Google Logo"
  className="w-6 h-6 mr-2 rounded-full"
/> Sign in with Google
              </button>
  )
}
