import React, { useState } from "react";
import logoImg from "../assets/logo.png";
import { registerUser, loginUser } from "../firebase/Auth";
import "./LoginPage.css";

const DESIGNATIONS = [
  "Research Scientist", "Senior Scientist", "Principal Scientist",
  "Research Associate", "Postdoctoral Researcher", "PhD Candidate",
  "Professor / Faculty", "Government Health Official", "Clinical Researcher",
  "Bioinformatician", "Pharmacologist", "Other"
];

const COUNTRIES = [
  "India", "United States", "United Kingdom", "Germany", "France",
  "Canada", "Australia", "Japan", "China", "Singapore", "Other"
];

export default function LoginPage({ onAuthSuccess }) {
  const [tab, setTab] = useState("login");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showId, setShowId] = useState(null); // show generated SCI ID after register

  const [loginForm, setLoginForm] = useState({ email: "", password: "" });
  const [regForm, setRegForm] = useState({
    name: "", email: "", password: "", confirm: "",
    designation: "", institution: "", department: "", country: ""
  });

  // ── LOGIN ──
  async function handleLogin() {
    setError("");
    if (!loginForm.email || !loginForm.password) { setError("Please fill in all fields."); return; }
    setLoading(true);
    try {
      await loginUser(loginForm.email, loginForm.password);
      onAuthSuccess();
    } catch (e) {
      setError(e.code === "auth/invalid-credential"
        ? "Invalid email or password."
        : e.message);
    } finally { setLoading(false); }
  }

  // ── REGISTER ──
  async function handleRegister() {
    setError("");
    const { name, email, password, confirm, designation, institution, department, country } = regForm;
    if (!name || !email || !password || !designation || !institution || !country) {
      setError("Please fill in all required fields."); return;
    }
    if (password !== confirm) { setError("Passwords do not match."); return; }
    if (password.length < 6) { setError("Password must be at least 6 characters."); return; }
    setLoading(true);
    try {
      const { scientistId } = await registerUser({ name, email, password, designation, institution, department, country });
      setShowId(scientistId);
    } catch (e) {
      setError(e.code === "auth/email-already-in-use"
        ? "This email is already registered."
        : e.message);
    } finally { setLoading(false); }
  }

  // ── SHOW GENERATED ID ──
  if (showId) {
    return (
      <div className="lp-root">
        <div className="lp-bg" />
        <div className="lp-card id-reveal">
          <img src={logoImg} alt="QureAI" className="lp-logo-img" />
          <h2>Registration Successful!</h2>
          <p className="id-desc">Your unique Scientist ID has been generated.<br />Save this — you'll need it for official records.</p>
          <div className="sci-id-box">
            <span className="sci-id-label">Your Scientist ID</span>
            <span className="sci-id-value">{showId}</span>
          </div>
          <p className="id-note">This ID is also stored in your profile and can be accessed anytime after login.</p>
          <button className="lp-btn" onClick={onAuthSuccess}>Enter QureAI →</button>
        </div>
      </div>
    );
  }

  return (
    <div className="lp-root">
      <div className="lp-bg" />
      <div className="lp-left">
        <div className="lp-brand">
          <img src={logoImg} alt="QureAI" className="lp-logo-img" />
          <span className="lp-brand-name">Qure<b>AI</b></span>
        </div>
        <h1 className="lp-tagline">
          Quantum-Powered<br />
          <span>Drug Discovery</span><br />
          for Rare Diseases
        </h1>
        <p className="lp-sub">
          A restricted research platform for verified scientists and government health officials.
          Login with your institutional credentials to access the QML pipeline.
        </p>
        
      </div>

      <div className="lp-right">
        <div className="lp-card">
          <div className="lp-card-header">
            <img src={logoImg} alt="QureAI" className="lp-logo-sm" />
            <div>
              <h3>Research Portal</h3>
              <p>Verified researchers & officials only</p>
            </div>
          </div>

          <div className="lp-tabs">
            <button className={tab === "login" ? "active" : ""} onClick={() => { setTab("login"); setError(""); }}>Login</button>
            <button className={tab === "register" ? "active" : ""} onClick={() => { setTab("register"); setError(""); }}>Register</button>
          </div>

          {tab === "login" ? (
            <div className="lp-form">
              <div className="lp-field">
                <label>Institutional Email</label>
                <input type="email" placeholder="you@research.org"
                  value={loginForm.email}
                  onChange={e => setLoginForm({ ...loginForm, email: e.target.value })}
                  onKeyDown={e => e.key === "Enter" && handleLogin()}
                />
              </div>
              <div className="lp-field">
                <label>Password</label>
                <input type="password" placeholder="••••••••"
                  value={loginForm.password}
                  onChange={e => setLoginForm({ ...loginForm, password: e.target.value })}
                  onKeyDown={e => e.key === "Enter" && handleLogin()}
                />
              </div>
              {error && <div className="lp-error">{error}</div>}
              <button className="lp-btn" onClick={handleLogin} disabled={loading}>
                {loading ? <span className="spinner" /> : "Access Research Portal →"}
              </button>
              <p className="lp-switch">New researcher? <span onClick={() => { setTab("register"); setError(""); }}>Register here</span></p>
            </div>
          ) : (
            <div className="lp-form">
              <div className="lp-field-row">
                <div className="lp-field">
                  <label>Full Name *</label>
                  <input type="text" placeholder="Dr. Jane Smith"
                    value={regForm.name} onChange={e => setRegForm({ ...regForm, name: e.target.value })} />
                </div>
                <div className="lp-field">
                  <label>Country *</label>
                  <select value={regForm.country} onChange={e => setRegForm({ ...regForm, country: e.target.value })}>
                    <option value="">Select</option>
                    {COUNTRIES.map(c => <option key={c}>{c}</option>)}
                  </select>
                </div>
              </div>
              <div className="lp-field">
                <label>Institutional Email *</label>
                <input type="email" placeholder="you@university.edu"
                  value={regForm.email} onChange={e => setRegForm({ ...regForm, email: e.target.value })} />
              </div>
              <div className="lp-field">
                <label>Designation / Title *</label>
                <select value={regForm.designation} onChange={e => setRegForm({ ...regForm, designation: e.target.value })}>
                  <option value="">Select your role</option>
                  {DESIGNATIONS.map(d => <option key={d}>{d}</option>)}
                </select>
              </div>
              <div className="lp-field">
                <label>Institution / Organization *</label>
                <input type="text" placeholder="MIT, AIIMS, WHO, etc."
                  value={regForm.institution} onChange={e => setRegForm({ ...regForm, institution: e.target.value })} />
              </div>
              <div className="lp-field">
                <label>Department <span className="optional">(optional)</span></label>
                <input type="text" placeholder="e.g. Computational Biology"
                  value={regForm.department} onChange={e => setRegForm({ ...regForm, department: e.target.value })} />
              </div>
              <div className="lp-field-row">
                <div className="lp-field">
                  <label>Password *</label>
                  <input type="password" placeholder="Min. 6 characters"
                    value={regForm.password} onChange={e => setRegForm({ ...regForm, password: e.target.value })} />
                </div>
                <div className="lp-field">
                  <label>Confirm Password *</label>
                  <input type="password" placeholder="Repeat password"
                    value={regForm.confirm} onChange={e => setRegForm({ ...regForm, confirm: e.target.value })} />
                </div>
              </div>
              <div className="lp-sci-hint">
                🔬 A unique <strong>Scientist ID (SCI-XXXXX)</strong> will be auto-generated upon registration.
              </div>
              {error && <div className="lp-error">{error}</div>}
              <button className="lp-btn" onClick={handleRegister} disabled={loading}>
                {loading ? <span className="spinner" /> : "Register & Get Scientist ID →"}
              </button>
              <p className="lp-switch">Already registered? <span onClick={() => { setTab("login"); setError(""); }}>Login here</span></p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}