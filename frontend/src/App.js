import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';
import { Component as GradientBg } from './components/ui/bg-gradient';
import AIAssistant from './components/ui/ai-assistant';

// Use Render backend URL in production, localhost in development
const API_URL = process.env.REACT_APP_API_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'https://fialerts.onrender.com'
    : 'http://localhost:8000');

const parseJwtPayload = (token) => {
  if (!token) return null;
  try {
    const base64Url = token.split('.')[1];
    if (!base64Url) return null;
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), '=');
    return JSON.parse(window.atob(padded));
  } catch {
    return null;
  }
};

function App() {
  const [apiKey, setApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [cachedTicker, setCachedTicker] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('fialerts_token') || null);
  const [userEmail, setUserEmail] = useState('');
  const [authView, setAuthView] = useState('login'); // 'login' | 'register' | 'apikey'

  // Login form state
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');

  // Register form state
  const [regEmail, setRegEmail] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regConfirm, setRegConfirm] = useState('');
  const [regFirst, setRegFirst] = useState('');
  const [regLast, setRegLast] = useState('');

  const [authLoading, setAuthLoading] = useState(false);
  const [authBootstrapLoading, setAuthBootstrapLoading] = useState(Boolean(localStorage.getItem('fialerts_token')));
  const [showCreateAlert, setShowCreateAlert] = useState(false);

  // On mount, if we have a saved token, try to load saved API key
  useEffect(() => {
    if (!token) {
      setAuthBootstrapLoading(false);
      return;
    }

    const loadSavedKey = async () => {
      try {
        const res = await axios.get(`${API_URL}/user/api-key`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (res?.data?.apiKey) {
          setApiKey(res.data.apiKey);
          setIsAuthenticated(true);
        }
      } catch (err) {
        if (err?.response?.status === 401 || err?.response?.status === 403) {
          // Token expired
          localStorage.removeItem('fialerts_token');
          setToken(null);
          return;
        }
        if (err?.response?.status === 404) {
          // No key saved yet — show API key entry
          setAuthView('apikey');
        }
      } finally {
        setAuthBootstrapLoading(false);
      }
    };
    loadSavedKey();
  }, [token]);

  // Extract email from token (JWT payload)
  useEffect(() => {
    if (!token) {
      setUserEmail('');
      return;
    }

    const payload = parseJwtPayload(token);
    setUserEmail(payload?.sub || '');
  }, [token]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setAuthLoading(true);
    try {
      const res = await axios.post(`${API_URL}/auth/login`, {
        email: loginEmail,
        password: loginPassword
      });
      const jwt = res.data.token;
      localStorage.setItem('fialerts_token', jwt);
      setUserEmail(res.data.email || '');
      setToken(jwt);
      setError('');
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.message || 'Login failed. Check your credentials.';
      setError(msg);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    if (regPassword !== regConfirm) { setError('Passwords do not match'); return; }
    if (regPassword.length < 6) { setError('Password must be at least 6 characters'); return; }
    setAuthLoading(true);
    try {
      const res = await axios.post(`${API_URL}/auth/register`, {
        email: regEmail,
        password: regPassword,
        firstName: regFirst,
        lastName: regLast
      });
      const jwt = res.data.token;
      localStorage.setItem('fialerts_token', jwt);
      setUserEmail(res.data.email || '');
      setToken(jwt);
      setError('');
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.message || 'Registration failed.';
      setError(msg);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleApiKeySubmit = async (e) => {
    e.preventDefault();
    if (!tempApiKey.trim()) { setError('Please enter your Google API key'); return; }
    try {
      if (token) {
        await axios.put(
          `${API_URL}/user/api-key`,
          { apiKey: tempApiKey },
          { headers: { Authorization: `Bearer ${token}` } }
        );
      }
      setApiKey(tempApiKey);
      setTempApiKey('');
      setIsAuthenticated(true);
      setError('');
    } catch (err) {
      console.error('Failed to save API key:', err);
      setError('Could not save your API key. Please try again.');
    }
  };

  const handleSendMessage = async (message) => {
    try {
      const result = await axios.post(`${API_URL}/analyze`, {
        message: message,
        api_key: apiKey,
        session_id: sessionId
      });
      setSessionId(result.data.session_id);
      setCachedTicker(result.data.cached_ticker);
      return {
        response: result.data.response,
        monitor_suggestion: result.data.monitor_suggestion || null,
      };
    } catch (err) {
      throw new Error(err.response?.data?.detail || 'An error occurred while analyzing. Please try again.');
    }
  };

  const handleCreateMonitor = async (monitor) => {
    // Auto-attach user email if not provided
    const payload = { ...monitor };
    if (!payload.user_email && userEmail) {
      payload.user_email = userEmail;
    }
    await axios.post(`${API_URL}/monitor`, payload);
  };

  const handleClearSession = () => {
    setSessionId(null);
    setCachedTicker(null);
  };

  const handleLogout = () => {
    localStorage.removeItem('fialerts_token');
    setToken(null);
    setAuthBootstrapLoading(false);
    setIsAuthenticated(false);
    setApiKey('');
    setTempApiKey('');
    setError('');
    setSessionId(null);
    setCachedTicker(null);
    setUserEmail('');
    setAuthView('login');
  };

  const handleChangeApiKey = async () => {
    try {
      if (token) {
        await axios.delete(`${API_URL}/user/api-key`, {
          headers: { Authorization: `Bearer ${token}` }
        });
      }
    } catch (err) {
      console.error('Failed to delete saved API key:', err);
    }
    setIsAuthenticated(false);
    setApiKey('');
    setTempApiKey('');
    setAuthView('apikey');
  };

  if (authBootstrapLoading) {
    return (
      <div className="App">
        <GradientBg gradientFrom="#ffffff" gradientTo="#e0e7ff" gradientPosition="50% 0%" gradientStop="40%" />
        <div className="fi-auth-container">
          <div className="fi-auth-card fi-auth-card--loading">
            <h2>FiAlerts.aio</h2>
            <p className="fi-auth-subtitle">Loading your account...</p>
          </div>
        </div>
      </div>
    );
  }

  // ===== AUTH SCREENS =====
  // Show login/register if no token
  if (!token) {
    return (
      <div className="App">
        <GradientBg gradientFrom="#ffffff" gradientTo="#e0e7ff" gradientPosition="50% 0%" gradientStop="40%" />
        <div className="fi-auth-container">
          <div className="fi-auth-card">
            <h2>FiAlerts.aio</h2>
            <p className="fi-auth-subtitle">AI-Powered Financial Analysis Agent</p>

            {error && <div className="fi-auth-error">{error}</div>}

            {authView === 'login' ? (
              <form onSubmit={handleLogin} className="fi-auth-form">
                <div className="fi-form-group">
                  <label htmlFor="loginEmail">Email</label>
                  <input id="loginEmail" type="email" placeholder="your@email.com" value={loginEmail} onChange={e => setLoginEmail(e.target.value)} required autoFocus />
                </div>
                <div className="fi-form-group">
                  <label htmlFor="loginPassword">Password</label>
                  <input id="loginPassword" type="password" placeholder="Enter your password" value={loginPassword} onChange={e => setLoginPassword(e.target.value)} required />
                </div>
                <button type="submit" className="fi-auth-button" disabled={authLoading}>
                  {authLoading ? 'Signing in...' : 'Sign In'}
                </button>
                <p className="fi-auth-footer">
                  Don't have an account?{' '}
                  <button type="button" className="fi-auth-link" onClick={() => { setAuthView('register'); setError(''); }}>Sign up</button>
                </p>
              </form>
            ) : (
              <form onSubmit={handleRegister} className="fi-auth-form">
                <div className="fi-form-row">
                  <div className="fi-form-group">
                    <label htmlFor="regFirst">First Name</label>
                    <input id="regFirst" type="text" placeholder="John" value={regFirst} onChange={e => setRegFirst(e.target.value)} required autoFocus />
                  </div>
                  <div className="fi-form-group">
                    <label htmlFor="regLast">Last Name</label>
                    <input id="regLast" type="text" placeholder="Doe" value={regLast} onChange={e => setRegLast(e.target.value)} required />
                  </div>
                </div>
                <div className="fi-form-group">
                  <label htmlFor="regEmail">Email</label>
                  <input id="regEmail" type="email" placeholder="your@email.com" value={regEmail} onChange={e => setRegEmail(e.target.value)} required />
                </div>
                <div className="fi-form-group">
                  <label htmlFor="regPassword">Password</label>
                  <input id="regPassword" type="password" placeholder="At least 6 characters" value={regPassword} onChange={e => setRegPassword(e.target.value)} required />
                </div>
                <div className="fi-form-group">
                  <label htmlFor="regConfirm">Confirm Password</label>
                  <input id="regConfirm" type="password" placeholder="Re-enter your password" value={regConfirm} onChange={e => setRegConfirm(e.target.value)} required />
                </div>
                <button type="submit" className="fi-auth-button" disabled={authLoading}>
                  {authLoading ? 'Creating account...' : 'Create Account'}
                </button>
                <p className="fi-auth-footer">
                  Already have an account?{' '}
                  <button type="button" className="fi-auth-link" onClick={() => { setAuthView('login'); setError(''); }}>Sign in</button>
                </p>
              </form>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Show API key entry if logged in but no API key saved
  if (!isAuthenticated) {
    return (
      <div className="App">
        <GradientBg gradientFrom="#ffffff" gradientTo="#e0e7ff" gradientPosition="50% 0%" gradientStop="40%" />
        <div className="fi-auth-container">
          <div className="fi-auth-card">
            <h2>FiAlerts.aio</h2>
            <p className="fi-auth-subtitle">One more step — enter your Google API key</p>

            {error && <div className="fi-auth-error">{error}</div>}

            <form onSubmit={handleApiKeySubmit} className="fi-auth-form">
              <div className="fi-form-group">
                <label htmlFor="apiKeyInput">Google Gemini API Key</label>
                <input
                  id="apiKeyInput" type="password"
                  placeholder="Paste your API key here"
                  value={tempApiKey}
                  onChange={e => setTempApiKey(e.target.value)}
                  autoFocus
                />
                <small className="fi-help-text">
                  Get your key from{' '}
                  <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a>
                </small>
              </div>
              <button type="submit" className="fi-auth-button">Save & Continue</button>
            </form>
            <p className="fi-auth-footer">
              <button type="button" className="fi-auth-link" onClick={handleLogout}>Sign out</button>
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ===== MAIN CHAT PAGE =====
  return (
    <div className="App">
      <GradientBg gradientFrom="#ffffff" gradientTo="#c7d2fe" gradientPosition="50% 0%" gradientStop="40%" />

      <header className="App-header">
        <h1>FiAlerts</h1>
        <p>AI-Powered Financial Analysis Agent</p>
        <div className="header-actions">
          <button className="logout-button" onClick={handleChangeApiKey}>Change API Key</button>
          <button className="clear-button" onClick={() => setShowCreateAlert(true)}>Create Alert</button>
          {cachedTicker && (
            <button className="clear-button" onClick={handleClearSession} title="Start fresh conversation">
              Clear Session ({cachedTicker})
            </button>
          )}
          <button className="logout-button" onClick={handleLogout}>Sign Out</button>
        </div>
      </header>

      <div className="container-modern">
        <AIAssistant
          apiKey={apiKey}
          onSendMessage={handleSendMessage}
          onCreateMonitor={handleCreateMonitor}
          openAlert={showCreateAlert}
          onCloseAlert={() => setShowCreateAlert(false)}
          userEmail={userEmail}
        />
      </div>

      <footer className="footer">
        <p>Powered by Google Gemini AI & yfinance</p>
      </footer>
    </div>
  );
}

export default App;
