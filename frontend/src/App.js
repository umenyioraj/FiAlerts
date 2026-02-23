import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [apiKey, setApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [cachedTicker, setCachedTicker] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!message.trim()) {
      setError('Please enter a question');
      return;
    }

    const userMessage = message.trim();
    
    // Add user message to conversation
    setConversation(prev => [...prev, { role: 'user', content: userMessage }]);
    
    setLoading(true);
    setError('');
    setMessage('');

    try {
      const result = await axios.post('http://localhost:8000/analyze', {
        message: userMessage,
        api_key: apiKey,
        session_id: sessionId
      });

      console.log('Analysis result:', result.data);
      console.log('Response:', result.data.response);
      console.log('Session ID:', result.data.session_id);
      console.log('Cached ticker:', result.data.cached_ticker);
      
      // Add AI response to conversation
      setConversation(prev => [...prev, { 
        role: 'assistant', 
        content: result.data.response,
        agents: result.data.agents_used 
      }]);
      
      setSessionId(result.data.session_id);
      setCachedTicker(result.data.cached_ticker);
      
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while analyzing. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const exampleQuestions = [
    "What stocks are good for long-term growth?",
    "Analyze AAPL stock",
    "Compare MSFT and GOOGL",
    "Find tech stocks with strong fundamentals",
    "What are the best dividend stocks?"
  ];

  const handleExampleClick = (question) => {
    setMessage(question);
  };

  // API Key Entry Page
  if (!isAuthenticated) {
    return (
      <div className="App">
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

      <header className="App-header">
        <h1> FiAlerts</h1>
        <p>AI-Powered Financial Analysis Agent</p>
        <div className="header-actions">
          <button 
            className="logout-button"
            onClick={() => {
              setIsAuthenticated(false);
              setTempApiKey('');
              setApiKey('');
              setMessage('');
              setConversation([]);
              setError('');
              setSessionId(null);
              setCachedTicker(null);
            }}
          >
            Change API Key
          </button>
          {cachedTicker && (
            <button 
              className="clear-button"
              onClick={() => {
                setSessionId(null);
                setCachedTicker(null);
                setConversation([]);
                setMessage('');
                setError('');
              }}
              title="Start fresh conversation"
            >
              Clear Session ({cachedTicker})
            </button>
          )}
        </div>
      </header>

      <div className="container">
        {/* Chat Messages Area */}
        <div className="chat-container">
          {conversation.length === 0 && !loading && (
            <div className="welcome-message">
              <h2>Ask me anything about stocks, financial analysis, or market trends!</h2>
              <div className="examples">
                <p className="examples-title">Try these examples:</p>
                <div className="example-buttons">
                  {exampleQuestions.map((question, index) => (
                    <button
                      key={index}
                      onClick={() => handleExampleClick(question)}
                      className="example-button"
                      type="button"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {conversation.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="message-content">
               
                <div className="message-text">
                  {msg.content.split('\n').map((line, i) => (
                    <p key={i} dangerouslySetInnerHTML={{ 
                      __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') || '&nbsp;' 
                    }} />
                  ))}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="loading-text">
              AI agents are analyzing your request...
            </div>
          )}

          {error && (
            <div className="error-box">
              <strong>Error:</strong> {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area (Fixed at Bottom) */}
        <div className="input-container">
          <form onSubmit={handleSubmit} className="chat-input-form">
            <textarea
              rows="2"
              placeholder="Ask a question about stocks..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              className="chat-input"
              disabled={loading}
            />
            <button type="submit" disabled={loading || !message.trim()} className={`send-button ${loading ? 'loading' : ''}`}>
              <span className="arrow"></span>
            </button>
          </form>
        </div>
      </div>

      <footer className="footer">
        <p>Powered by Google Gemini AI & yfinance</p>
      </footer>
    </div>
  );
}

export default App;
