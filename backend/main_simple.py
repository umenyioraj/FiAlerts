from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
import pandas as pd
import uuid
from datetime import datetime, timedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
import yfinance as yf

app = FastAPI(title="FiAlerts API - Simple")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://fialerts.netlify.app"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage for conversation memory and cached stock data
# Format: {session_id: {"ticker": str, "data": dict, "timestamp": datetime, "history": list}}
SESSION_STORE = {}
SESSION_TIMEOUT = timedelta(hours=1)  # Sessions expire after 1 hour

def round_numbers(obj, decimals=2):
    """Recursively round all numeric values in a dictionary or list."""
    if isinstance(obj, dict):
        return {k: round_numbers(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_numbers(item, decimals) for item in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    else:
        return obj

class QueryRequest(BaseModel):
    message: str
    api_key: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    agents_used: list[str]
    session_id: str
    cached_ticker: Optional[str] = None

# ============= FINANCIAL TOOLS =============
@tool
def get_stock_price(ticker: str) -> dict:
    """Get current stock price and basic info for a given ticker symbol."""
    try:
        stock = yf.Ticker(ticker.upper())
        return {
            "ticker": ticker.upper(),
            "current_price": stock.fast_info['lastPrice'],
            "previous_close": stock.fast_info['previousClose'],
            "day_high": stock.fast_info['dayHigh'],
            "day_low": stock.fast_info['dayLow'],
            "volume": stock.fast_info['lastVolume']
        }
    except Exception as e:
        return {"error": f"Could not fetch data for {ticker}: {str(e)}"}

@tool
def get_financial_metrics(ticker: str) -> dict:
    """Get comprehensive financial metrics: P/E ratio, P/B ratio, ROE, debt-to-equity, profit margins, etc."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName"),
            
            # Valuation Ratios
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            
            # Profitability Metrics
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            
            # Debt & Liquidity
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            
            # Growth & Dividends
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            
            # Other Key Metrics
            "beta": info.get("beta"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "shares_outstanding": info.get("sharesOutstanding")
        }
    except Exception as e:
        return {"error": f"Could not fetch metrics for {ticker}: {str(e)}"}

@tool
def get_technical_indicators(ticker: str, period: str = "3mo") -> dict:
    """Get technical indicators including RSI, MACD, EMA crossovers, and moving averages."""
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        
        if hist.empty or len(hist) < 50:
            return {"error": f"Insufficient historical data for {ticker}"}
        
        # Simple Moving Averages
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        hist['EMA_9'] = hist['Close'].ewm(span=9, adjust=False).mean()
        hist['EMA_21'] = hist['Close'].ewm(span=21, adjust=False).mean()
        
        # RSI Calculation (14-period)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD Calculation (12, 26, 9)
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = exp1 - exp2
        hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACD_Histogram'] = hist['MACD'] - hist['MACD_Signal']
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        # Determine EMA crossover
        ema_crossover = None
        if not pd.isna(latest['EMA_9']) and not pd.isna(latest['EMA_21']):
            if latest['EMA_9'] > latest['EMA_21'] and prev['EMA_9'] <= prev['EMA_21']:
                ema_crossover = "bullish_crossover"
            elif latest['EMA_9'] < latest['EMA_21'] and prev['EMA_9'] >= prev['EMA_21']:
                ema_crossover = "bearish_crossover"
            elif latest['EMA_9'] > latest['EMA_21']:
                ema_crossover = "bullish"
            else:
                ema_crossover = "bearish"
        
        # MACD Signal
        macd_signal = None
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                macd_signal = "bullish"
            else:
                macd_signal = "bearish"
        
        return {
            "ticker": ticker.upper(),
            "current_price": float(latest['Close']),
            "sma_20": float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
            "sma_50": float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
            "ema_9": float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None,
            "ema_21": float(latest['EMA_21']) if not pd.isna(latest['EMA_21']) else None,
            "ema_crossover_status": ema_crossover,
            "rsi": float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
            "rsi_signal": "overbought" if not pd.isna(latest['RSI']) and latest['RSI'] > 70 else "oversold" if not pd.isna(latest['RSI']) and latest['RSI'] < 30 else "neutral",
            "macd": float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
            "macd_signal_line": float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else None,
            "macd_histogram": float(latest['MACD_Histogram']) if not pd.isna(latest['MACD_Histogram']) else None,
            "macd_trend": macd_signal,
            "volume": int(latest['Volume']),
            "trend": "bullish" if latest['Close'] > prev['Close'] else "bearish"
        }
    except Exception as e:
        return {"error": f"Could not calculate indicators for {ticker}: {str(e)}"}

@tool
def get_news_sentiment(ticker: str) -> dict:
    """Get recent news and sentiment for a stock ticker."""
    try:
        stock = yf.Ticker(ticker.upper())
        news = stock.news[:5] if stock.news else []
        
        # Simple sentiment based on news titles
        sentiment_score = 0
        positive_words = ['surge', 'jump', 'rally', 'gain', 'rise', 'beat', 'strong', 'growth', 'up', 'high', 'profit']
        negative_words = ['fall', 'drop', 'decline', 'loss', 'weak', 'miss', 'concern', 'risk', 'down', 'low', 'cut']
        
        for article in news:
            title = article.get('title', '').lower()
            sentiment_score += sum(1 for word in positive_words if word in title)
            sentiment_score -= sum(1 for word in negative_words if word in title)
        
        sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        
        return {
            "ticker": ticker.upper(),
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "news_count": len(news),
            "recent_headlines": [n.get('title') for n in news[:3]]
        }
    except Exception as e:
        return {"error": f"Could not fetch sentiment for {ticker}: {str(e)}"}

@tool
def get_stock_universe(category: str = "large_cap") -> dict:
    """Get a list of stocks to analyze. Categories: 'large_cap', 'tech', 'finance', 'healthcare', 'energy'."""
    stock_lists = {
        "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "JNJ"],
        "tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "ADBE", "CRM", "ORCL", "INTC"],
        "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
        "healthcare": ["JNJ", "UNH", "PFE", "LLY", "ABBV", "TMO", "MRK", "ABT", "DHR", "BMY"],
        "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
        "consumer": ["AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "COST"],
        "dividend": ["JNJ", "PG", "KO", "PEP", "MCD", "WMT", "VZ", "T", "XOM", "CVX"],
        "indexes" : ["SMH", "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"]
    }
    
    tickers = stock_lists.get(category.lower(), stock_lists["large_cap"])
    
    return {
        "category": category,
        "tickers": tickers,
        "count": len(tickers),
        "description": f"List of {category} stocks for analysis"
    }

@tool
def screen_stocks_for_growth(tickers_str: str, max_stocks: int = 1) -> dict:
    """Screen multiple stocks for long-term growth potential. Pass tickers as comma-separated string. Returns only the top 1 stock by default to save API quota."""
    try:
        tickers = [t.strip().upper() for t in tickers_str.split(",")]
        results = []
        
        for ticker in tickers[:10]:  # Limit to 10 stocks to avoid timeout
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Key growth metrics
                pe_ratio = info.get("trailingPE")
                peg_ratio = info.get("pegRatio")
                revenue_growth = info.get("revenueGrowth")
                earnings_growth = info.get("earningsGrowth")
                roe = info.get("returnOnEquity")
                debt_to_equity = info.get("debtToEquity")
                profit_margin = info.get("profitMargins")
                
                # Growth score calculation (0-100)
                score = 50
                if revenue_growth and revenue_growth > 0.15: score += 15
                if earnings_growth and earnings_growth > 0.15: score += 15
                if roe and roe > 0.15: score += 10
                if profit_margin and profit_margin > 0.15: score += 10
                if pe_ratio and 15 < pe_ratio < 30: score += 10
                elif pe_ratio and pe_ratio < 15: score += 5
                elif pe_ratio and pe_ratio > 40: score -= 10
                if peg_ratio and peg_ratio < 1.5: score += 10
                elif peg_ratio and peg_ratio > 3: score -= 5
                if debt_to_equity and debt_to_equity < 50: score += 5
                elif debt_to_equity and debt_to_equity > 150: score -= 10
                
                results.append({
                    "ticker": ticker,
                    "company_name": info.get("longName", "N/A"),
                    "growth_score": max(0, min(100, score)),
                    "pe_ratio": pe_ratio,
                    "peg_ratio": peg_ratio,
                    "revenue_growth": f"{revenue_growth*100:.1f}%" if revenue_growth else "N/A",
                    "roe": f"{roe*100:.1f}%" if roe else "N/A"
                })
            except:
                continue
        
        # Sort by growth score and return only top stocks
        results.sort(key=lambda x: x.get("growth_score", 0), reverse=True)
        top_results = results[:max_stocks]
        
        return {
            "screened_stocks": top_results,
            "top_pick": top_results[0]["ticker"] if top_results else None,
            "top_tickers": [r["ticker"] for r in top_results]
        }
    except Exception as e:
        return {"error": f"Screening error: {str(e)}"}

# ============= SINGLE AGENT SYSTEM =============
class AgentState(TypedDict):
    messages: list
    iteration_count: int

# Combine all tools
all_tools = [
    get_stock_price, 
    get_financial_metrics, 
    get_technical_indicators, 
    get_news_sentiment,
    get_stock_universe,
    screen_stocks_for_growth
]

# ============= SESSION MANAGEMENT =============
def clean_expired_sessions():
    """Remove sessions older than SESSION_TIMEOUT."""
    now = datetime.now()
    expired = [sid for sid, data in SESSION_STORE.items() 
               if now - data.get("timestamp", now) > SESSION_TIMEOUT]
    for sid in expired:
        del SESSION_STORE[sid]

def get_or_create_session(session_id: Optional[str]) -> str:
    """Get existing session or create new one."""
    clean_expired_sessions()
    
    if session_id and session_id in SESSION_STORE:
        # Update timestamp
        SESSION_STORE[session_id]["timestamp"] = datetime.now()
        return session_id
    
    # Create new session
    new_id = str(uuid.uuid4())
    SESSION_STORE[new_id] = {
        "ticker": None,
        "data": {},
        "timestamp": datetime.now(),
        "history": []
    }
    return new_id

def cache_stock_data(session_id: str, ticker: str, financial_data: dict, technical_data: dict, sentiment_data: dict):
    """Cache stock data for a session."""
    if session_id in SESSION_STORE:
        SESSION_STORE[session_id]["ticker"] = ticker
        SESSION_STORE[session_id]["data"] = {
            "financial": financial_data,
            "technical": technical_data,
            "sentiment": sentiment_data
        }
        SESSION_STORE[session_id]["timestamp"] = datetime.now()

def get_cached_data(session_id: str) -> Optional[dict]:
    """Get cached stock data from session."""
    if session_id in SESSION_STORE:
        session = SESSION_STORE[session_id]
        if session.get("ticker") and session.get("data"):
            return {
                "ticker": session["ticker"],
                **session["data"]
            }
    return None

UNIFIED_AGENT_PROMPT = """You are a Financial Analysis AI Agent specializing in stock market analysis.

Your role is to analyze financial data and provide actionable investment insights.

Formatting Rules:
- Use **double asterisks** ONLY for critical numbers and key terms (ticker symbols, recommendations like BUY/HOLD/SELL)
- Use section headers like "1. Company Overview" without asterisks
- Write most content in plain text without formatting
- Be selective - only bold the most important data points

When you receive financial data for a stock (metrics, technical indicators, sentiment), provide a comprehensive analysis with:

1. **Company Overview**: Summarize the company and current stock price
2. **Financial Health**: Analyze valuation ratios (P/E, P/B, PEG), profitability margins, debt levels, and growth rates
3. **Technical Analysis**: Interpret RSI, MACD, moving averages, and trend signals
4. **Sentiment Analysis**: Summarize news sentiment and recent headlines
5. **Investment Recommendation**: Clear BUY/HOLD/SELL recommendation with specific reasoning based on the data

Be specific - reference actual numbers from the data. Avoid generic statements.

CRITICAL: If you see "FOLLOW-UP question" in the prompt, DO NOT provide the full 5-section analysis again. 
Only answer the specific question asked. Be concise and direct.

For stock screening requests ("best tech stocks", "dividend stocks", etc.):
1. Use get_stock_universe() to get relevant tickers
2. Use screen_stocks_for_growth() to find top candidates
3. Analyze the results and provide recommendations

When answering specific questions (buy/sell timing, price targets, strategy advice):
- Answer directly without repeating the full analysis
- Reference only the relevant metrics needed for the answer
- Be concise and actionable

If user provides their purchase price or dollar cost average (DCA):
- Incorporate that into your recommendation
- Compare current price to their entry point
- Suggest specific strategies only if asked

Always be direct and data-driven."""


def create_simple_workflow(api_key: str):
    """Create a single-agent workflow that uses fewer API calls."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.1
    )
    
    def agent(state: AgentState):
        messages = state["messages"]
        iteration_count = state.get("iteration_count", 0)
        
        # Force final response on last iteration
        if iteration_count >= 3:
            # On the 4th call, don't bind tools - force a text response
            valid_messages = []
            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    if msg.content and (isinstance(msg.content, str) or isinstance(msg.content, list)):
                        valid_messages.append(msg)
            
            valid_messages.append(HumanMessage(content="Based on all the tool results above, provide your complete financial analysis of the stock now. Include key findings from the data."))
            response = llm.invoke(valid_messages)  # No tools bound - text only
            return {"messages": [response], "iteration_count": iteration_count + 1}
        
        # If this is the first call, add the system prompt
        if iteration_count == 0:
            user_content = messages[0].content
            
            # Check if data has already been provided (ticker analysis)
            if "FINANCIAL METRICS:" in user_content and "TECHNICAL INDICATORS:" in user_content:
                # Data is already provided, agent should just analyze
                first_msg = HumanMessage(content=f"{UNIFIED_AGENT_PROMPT}\n\n{user_content}")
                response = llm.invoke([first_msg])  # No tools needed, just analysis
            else:
                # General query, agent can use tools if needed
                first_msg = HumanMessage(content=f"{UNIFIED_AGENT_PROMPT}\n\nUser: {user_content}")
                response = llm.bind_tools(all_tools).invoke([first_msg])
        else:
            # Subsequent calls - filter and format messages properly
            # Only keep HumanMessage and AIMessage with valid content
            valid_messages = []
            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    if msg.content and (isinstance(msg.content, str) or isinstance(msg.content, list)):
                        valid_messages.append(msg)
            
            # Add instruction to continue
            valid_messages.append(HumanMessage(content="Based on the data above, provide your complete analysis now or use additional tools if absolutely necessary."))
            response = llm.bind_tools(all_tools).invoke(valid_messages)
        
        return {"messages": [response], "iteration_count": iteration_count + 1}
    
    def should_continue(state: AgentState):
        iteration_count = state.get("iteration_count", 0)
        
        # Force stop after 4 agent calls (4 API calls max)
        if iteration_count >= 4:
            return END
            
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    tool_node = ToolNode(all_tools)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ============= API ENDPOINTS =============
@app.post("/analyze", response_model=QueryResponse)
async def analyze_stock(request: QueryRequest):
    """Analyze a stock using single unified agent (fewer API calls)."""
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        
        # Extract ticker from natural language if present
        import re
        message = request.message

        question_words = {'WHAT', 'GOOD', 'WHEN', 'HOW', 'IS', 'ARE', 'DO', 'SHOULD', 'WOULD', 'COULD', 'CAN'}

        if message.split()[0].upper() in question_words:
            # If the message starts with a question word remove it from the possible tickers
            message = ' '.join(message.split()[1:])

        # Look for all potential stock ticker patterns (2-5 uppercase letters)
        potential_tickers = re.findall(r'\b([A-Z]{2,5})\b', message.upper())
        
        # Filter out common words that aren't tickers (including pronouns)
        excluded_words = {
            'STOCK', 'TICKER', 'ANALYZE', 'COMPARE', 'BUY', 'SELL', 'FIND', 'WHAT', 'THE', 'IS', 'ARE', 
            'AND', 'OR', 'FOR', 'WITH', 'ABOUT', 'LOOK', 'AT', 'VS', 'ETF', 'INDEX',
            'IT', 'IF', 'TO', 'IN', 'ON', 'MY', 'ME', 'WE', 'US', 'AS', 'BE', 'BY', 'AN', 'SO', 'UP', 'DOWN',
            'WHEN', 'WHERE', 'WHY', 'WHO', 'HAVE', 'HAS', 'HAD', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD',
            'SHOULD', 'MAY', 'MIGHT', 'MUST', 'CAN', 'THAN', 'THEN', 'THEM', 'THIS', 'THAT', 'THESE', 'THOSE',
            'FROM', 'INTO', 'OUT', 'OVER', 'UNDER', 'ABOVE', 'BELOW', 'AFTER', 'BEFORE', 'SINCE', 'UNTIL',
            'PRICE', 'SHARE', 'BOUGHT', 'MARKET', 'DCA', 'TRADE', 'YEAR', 'DAY', 'MONTH', 'WEEK', 'HIGH', 'LOW'
        }
        potential_tickers = [t for t in potential_tickers if t not in excluded_words]
        
        # Check if we have a cached ticker from previous conversation
        cached_data = get_cached_data(session_id)
        has_cached_ticker = cached_data and cached_data.get('ticker')
        
        # Validate each ticker with yfinance to find the real one
        ticker = None
        for candidate in potential_tickers:
            try:
                test_stock = yf.Ticker(candidate)
                # Try to get basic info - if it fails, ticker is invalid
                info = test_stock.info
                if info and info.get('symbol'):
                    ticker = candidate
                    break
            except Exception as e:
                continue
        
        # If we have a cached ticker but detected a new one, prefer cached for follow-up questions
        # This handles cases like "when should I sell IT" where IT might be detected as a ticker
        if has_cached_ticker and ticker and len(ticker) <= 2:
            # If detected ticker is 2 letters or less and we have cached data, it's likely a pronoun
            # Use cached ticker instead
            ticker = None
        
        enhanced_message = message
        cached_ticker = None
        
        # If we detected a valid ticker, call tools directly and ask agent to analyze
        if ticker:
            # Call the tools directly
            financial_data = round_numbers(get_financial_metrics.invoke({"ticker": ticker}))
            technical_data = round_numbers(get_technical_indicators.invoke({"ticker": ticker}))
            sentiment_data = get_news_sentiment.invoke({"ticker": ticker})
            
            # Cache the data for this session
            cache_stock_data(session_id, ticker, financial_data, technical_data, sentiment_data)
            cached_ticker = ticker
            
            # Format the data for the agent
            data_summary = f"""Here is the complete data for {ticker}:

FINANCIAL METRICS:
{financial_data}

TECHNICAL INDICATORS:
{technical_data}

NEWS SENTIMENT:
{sentiment_data}

Based on this data, provide a comprehensive financial analysis including:
1. Company overview and current valuation
2. Financial health assessment (profitability, debt, growth)
3. Technical analysis (trend, momentum indicators)
4. Sentiment analysis from news
5. Final recommendation (Buy/Hold/Sell) with clear reasoning

Be specific and reference the actual numbers from the data above."""
            
            enhanced_message = data_summary
        
        # If no ticker detected, check if we have cached data from previous conversation
        elif not ticker and cached_data:
            cached_ticker = cached_data["ticker"]
            # Round cached data in case it was cached before rounding was added
            rounded_financial = round_numbers(cached_data['financial'])
            rounded_technical = round_numbers(cached_data['technical'])
            
            # Use cached data to answer follow-up question
            data_summary = f"""IMPORTANT: This is a FOLLOW-UP question about {cached_ticker}. You already provided a full analysis earlier.

User's follow-up question: {request.message}

DO NOT provide the full analysis again. Just answer the specific question using the data below.

Reference data for {cached_ticker}:

FINANCIAL METRICS:
{rounded_financial}

TECHNICAL INDICATORS:
{rounded_technical}

NEWS SENTIMENT:
{cached_data['sentiment']}

Answer ONLY the user's specific question. Be concise and direct. Reference relevant numbers from the data."""
            
            enhanced_message = data_summary
        
        workflow = create_simple_workflow(request.api_key)
        
        initial_state = {
            "messages": [HumanMessage(content=enhanced_message)],
            "iteration_count": 0
        }
        
        # Limit to 20 steps - this allows multiple tool calls but prevents infinite loops
        # Under free tier limit of 20 API calls per day
        result = workflow.invoke(initial_state, {"recursion_limit": 20})
        
        # Collect AI responses
        responses = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, list):
                    text_parts = []
                    for item in msg.content:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                        else:
                            text_parts.append(str(item))
                    responses.append(" ".join(text_parts))
                else:
                    responses.append(msg.content)
        
        final_response = "\n\n".join(responses)
        
        if not final_response or final_response.strip() == "":
            final_response = "Analysis completed but no response was generated. Please try again with a different question."
        
        return QueryResponse(
            response=final_response,
            agents_used=["Unified Agent (Simple Mode)"],
            session_id=session_id,
            cached_ticker=cached_ticker
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "simple"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
