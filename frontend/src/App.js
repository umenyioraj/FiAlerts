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

const TRACKR_API_URL = (
  process.env.REACT_APP_TRACKR_API_URL ||
  (process.env.NODE_ENV === 'production'
    ? 'https://trackr-backend-lyot.onrender.com/api'
    : 'http://localhost:8080/api')
).trim();

const ALLOWED_PARENT_ORIGINS = new Set([
  'http://localhost:5173',
  'https://trackr-aio.netlify.app'
]);

const isAllowedParentOrigin = (origin) => {
  if (!origin) return false;
  if (origin === 'null') return true;
  if (ALLOWED_PARENT_ORIGINS.has(origin)) return true;
  return /^https:\/\/.*\.netlify\.app$/i.test(origin);
};

function App() {
  const [apiKey, setApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [cachedTicker, setCachedTicker] = useState(null);
  const [trackrToken, setTrackrToken] = useState(null);

  const requestTrackUiAuth = () => {
    try {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: 'REQUEST_TRACKUI_AUTH' }, '*');
        window.parent.postMessage({ type: 'REQUEST_TRACK_UI_CONTEXT' }, '*');
      }
    } catch {
      // no-op
    }
  };

  useEffect(() => {
    requestTrackUiAuth();
    try {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: 'FIALERTS_READY' }, '*');
      }
    } catch {
      // no-op
    }
  }, []);

  useEffect(() => {
    if (trackrToken) return;
    if (!(window.parent && window.parent !== window)) return;

    const retryId = window.setInterval(() => {
      requestTrackUiAuth();
    }, 1000);

    const stopId = window.setTimeout(() => {
      window.clearInterval(retryId);
    }, 10000);

    return () => {
      window.clearInterval(retryId);
      window.clearTimeout(stopId);
    };
  }, [trackrToken]);

  useEffect(() => {
    const onMessage = (event) => {
      if (!isAllowedParentOrigin(event.origin)) return;
      if (!event.data || event.data.type !== 'TRACKUI_AUTH') return;
      if (!event.data.token) return;

      setTrackrToken(event.data.token);
    };

    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, []);

  useEffect(() => {
    const loadSavedKey = async () => {
      if (!trackrToken) return;
      try {
        const res = await axios.get(`${TRACKR_API_URL}/user/google-api-key`, {
          headers: { Authorization: `Bearer ${trackrToken}` }
        });
        if (res?.data?.googleApiKey) {
          setApiKey(res.data.googleApiKey);
          setIsAuthenticated(true);
          setError('');
        }
      } catch (err) {
        if (err?.response?.status === 404) {
          // Key not saved yet
          return;
        }
        console.error('Failed to load saved API key:', err);
      }
    };

    loadSavedKey();
  }, [trackrToken]);

  const handleApiKeySubmit = async (e) => {
    e.preventDefault();
    
    if (!tempApiKey.trim()) {
      setError('Please enter your Google API key');
      return;
    }
    
    try {
      if (trackrToken) {
        await axios.put(
          `${TRACKR_API_URL}/user/google-api-key`,
          { googleApiKey: tempApiKey },
          { headers: { Authorization: `Bearer ${trackrToken}` } }
        );
      }

      setApiKey(tempApiKey);
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

      console.log('Analysis result:', result.data);
      
      setSessionId(result.data.session_id);
      setCachedTicker(result.data.cached_ticker);
      
      return result.data.response;
    } catch (err) {
      console.error('Error:', err);
      throw new Error(err.response?.data?.detail || 'An error occurred while analyzing. Please try again.');
    }
  };

  const handleClearSession = () => {
    setSessionId(null);
    setCachedTicker(null);
  };

  const handleLogout = async () => {
    try {
      if (trackrToken) {
        await axios.delete(`${TRACKR_API_URL}/user/google-api-key`, {
          headers: { Authorization: `Bearer ${trackrToken}` }
        });
      }
    } catch (err) {
      console.error('Failed to delete saved API key:', err);
    }

    setIsAuthenticated(false);
    setTempApiKey('');
    setApiKey('');
    setError('');
    setSessionId(null);
    setCachedTicker(null);
  };

  // API Key Entry Page
  if (!isAuthenticated) {
    return (
      <div className="App">
        <GradientBg 
          gradientFrom="#ffffff"
          gradientTo="#e0e7ff"
          gradientPosition="50% 0%"
          gradientStop="40%"
        />
        <div className="auth-container">
          <div className="auth-box">
            <div className="auth-header">
              <h1>FiAlerts.aio</h1>
              <p>AI-Powered Financial Analysis Agent</p>
            </div>

            <form onSubmit={handleApiKeySubmit} className="auth-form">
              <div className="form-group">
                <label htmlFor="apiKeyInput">Enter Your Google API Key</label>
                <input
                  id="apiKeyInput"
                  type="password"
                  placeholder="Paste your Google Gemini API key here"
                  value={tempApiKey}
                  onChange={(e) => setTempApiKey(e.target.value)}
                  className="input-field"
                  autoFocus
                />
                <small className="help-text">
                  Get your API key from{' '}
                  <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer">
                    Google AI Studio
                  </a>
                </small>
                {trackrToken && (
                  <small className="help-text">
                    Your key will be saved to your Track-UI account.
                  </small>
                )}
              </div>

              {error && (
                <div className="error-box">
                  {error}
                </div>
              )}

              <button type="submit" className="submit-button">
                Enter
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  // Main Chat Page
  return (
    <div className="App">
      <GradientBg 
        gradientFrom="#ffffff"
        gradientTo="#c7d2fe"
        gradientPosition="50% 0%"
        gradientStop="40%"
      />

      <header className="App-header">
        <h1>FiAlerts</h1>
        <p>AI-Powered Financial Analysis Agent</p>
        <div className="header-actions">
          <button 
            className="logout-button"
            onClick={handleLogout}
          >
            Change API Key
          </button>
          {cachedTicker && (
            <button 
              className="clear-button"
              onClick={handleClearSession}
              title="Start fresh conversation"
            >
              Clear Session ({cachedTicker})
            </button>
          )}
        </div>
      </header>

      <div className="container-modern">
        <AIAssistant 
          apiKey={apiKey}
          onSendMessage={handleSendMessage}
        />
      </div>

      <footer className="footer">
        <p>Powered by Google Gemini AI & yfinance</p>
      </footer>
    </div>
  );
}

export default App;
