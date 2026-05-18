import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged
} from "firebase/auth";
import { doc, setDoc, getDoc } from "firebase/firestore";
import { auth, db } from "./config";

// Generate a unique Scientist/Researcher ID
export function generateScientistId() {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let id = "SCI-";
  for (let i = 0; i < 5; i++) id += chars[Math.floor(Math.random() * chars.length)];
  return id;
}

// Register new user — stores profile in Firestore
export async function registerUser({ name, email, password, designation, institution, department, country }) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  const scientistId = generateScientistId();

  await setDoc(doc(db, "users", cred.user.uid), {
    uid: cred.user.uid,
    scientistId,
    name,
    email,
    designation,
    institution,
    department,
    country,
    createdAt: new Date().toISOString(),
    accessLevel: "Research Associate",
    activeProjects: 0,
  });

  return { user: cred.user, scientistId };
}

// Login existing user
export async function loginUser(email, password) {
  const cred = await signInWithEmailAndPassword(auth, email, password);
  return cred.user;
}

// Logout
export async function logoutUser() {
  await signOut(auth);
}

// Fetch user profile from Firestore
export async function fetchUserProfile(uid) {
  const snap = await getDoc(doc(db, "users", uid));
  return snap.exists() ? snap.data() : null;
}

// Auth state listener
export function onAuthChange(callback) {
  return onAuthStateChanged(auth, callback);
}