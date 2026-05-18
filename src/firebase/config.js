import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyA0sT07EWSkaJ14YJ7rH1KVd0TJ62A2MbY",
  authDomain: "qureai-e0322.firebaseapp.com",
  projectId: "qureai-e0322",
  storageBucket: "qureai-e0322.firebasestorage.app",
  messagingSenderId: "194637716798",
  appId: "1:194637716798:web:f7a6edc0ff3a562fab206f",
  measurementId: "G-FRRZZCZCKQ"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);