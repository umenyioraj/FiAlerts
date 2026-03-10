import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import { Component as GradientBg } from './components/ui/bg-gradient';
import AIAssistant from './components/ui/ai-assistant';

// Use Render backend URL in production, localhost in development
const API_URL = process.env.REACT_APP_API_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'https://fialerts.onrender.com'
    : 'http://localhost:8000');

function App() {
  const [apiKey, setApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [cachedTicker, setCachedTicker] = useState(null);

  const handleApiKeySubmit = (e) => {
    e.preventDefault();
    
    if (!tempApiKey.trim()) {
      setError('Please enter your Google API key');
      return;
    }
    
    setApiKey(tempApiKey);
    setIsAuthenticated(true);
    setError('');
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

  const handleLogout = () => {
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
