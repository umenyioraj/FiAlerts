# FiAlerts - AI Financial Analysis Agent

A multi-agent AI system for comprehensive stock market analysis powered by Google Gemini and yfinance.

## Features

- **Multi-Agent Architecture**: 4 specialized AI agents work together
  - **Screening Agent**: Finds and recommends stocks based on criteria
  - **Financial Agent**: Analyzes fundamentals (P/E, ROE, debt ratios, etc.)
  - **Technical Agent**: Analyzes technical indicators (RSI, MACD, EMA crossovers)
  - **Sentiment Agent**: Analyzes news and market sentiment

- **Smart Stock Screening**: Get recommendations for long-term growth, dividends, or sector-specific stocks
- **Real-time Market Data**: Powered by yfinance for accurate stock information
- **Interactive UI**: Modern React frontend with example questions

## Setup

### Backend Setup

1. Install Python dependencies:
```bash
cd FiAlerts
py -m pip install -r requirments.txt
```

2. Run the FastAPI backend:
```bash
cd FiAlerts
.venv\Scripts\python.exe main.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the React development server:
```bash
npm start
```

The frontend will open at `http://localhost:3000`

## Getting Your Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it into the frontend

## Example Questions

- "What stocks are good for long-term growth?"
- "Analyze AAPL stock"
- "Compare MSFT and GOOGL"
- "Find tech stocks with strong fundamentals"
- "What are the best dividend stocks?"

## Tech Stack

**Backend:**
- FastAPI
- LangChain & LangGraph
- Google Gemini AI
- yfinance
- pandas

**Frontend:**
- React
- Axios
- CSS3

## API Endpoint

**POST** `/analyze`

Request body:
```json
{
  "message": "What stocks are good for long-term growth?",
  "api_key": "your-google-api-key"
}
```

Response:
```json
{
  "response": "Based on analysis...",
  "agents_used": ["Screening Agent", "Financial Agent", "Technical Agent", "Sentiment Agent"]
}
```
