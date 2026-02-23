from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
import yfinance as yf

app = FastAPI(title="FiAlerts API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str
    api_key: str

class QueryResponse(BaseModel):
    response: str
    agents_used: list[str]

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
            "roe": info.get("returnOnEquity"),  # Return on Equity
            "roa": info.get("returnOnAssets"),  # Return on Assets
            
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
def get_income_statement(ticker: str) -> dict:
    """Get income statement data including revenue, earnings, EBITDA."""
    try:
        stock = yf.Ticker(ticker.upper())
        financials = stock.financials
        
        if financials.empty:
            return {"error": f"No financial data available for {ticker}"}
        
        latest = financials.iloc[:, 0]  # Most recent data
        
        return {
            "ticker": ticker.upper(),
            "period": str(financials.columns[0].date()),
            "total_revenue": int(latest.get('Total Revenue', 0)) if 'Total Revenue' in latest else None,
            "gross_profit": int(latest.get('Gross Profit', 0)) if 'Gross Profit' in latest else None,
            "operating_income": int(latest.get('Operating Income', 0)) if 'Operating Income' in latest else None,
            "ebitda": int(latest.get('EBITDA', 0)) if 'EBITDA' in latest else None,
            "net_income": int(latest.get('Net Income', 0)) if 'Net Income' in latest else None,
        }
    except Exception as e:
        return {"error": f"Could not fetch income statement for {ticker}: {str(e)}"}

@tool
def get_balance_sheet(ticker: str) -> dict:
    """Get balance sheet data including assets, liabilities, equity."""
    try:
        stock = yf.Ticker(ticker.upper())
        balance_sheet = stock.balance_sheet
        
        if balance_sheet.empty:
            return {"error": f"No balance sheet data available for {ticker}"}
        
        latest = balance_sheet.iloc[:, 0]  # Most recent data
        
        return {
            "ticker": ticker.upper(),
            "period": str(balance_sheet.columns[0].date()),
            "total_assets": int(latest.get('Total Assets', 0)) if 'Total Assets' in latest else None,
            "total_liabilities": int(latest.get('Total Liabilities Net Minority Interest', 0)) if 'Total Liabilities Net Minority Interest' in latest else None,
            "stockholders_equity": int(latest.get('Stockholders Equity', 0)) if 'Stockholders Equity' in latest else None,
            "total_debt": int(latest.get('Total Debt', 0)) if 'Total Debt' in latest else None,
            "cash_and_equivalents": int(latest.get('Cash And Cash Equivalents', 0)) if 'Cash And Cash Equivalents' in latest else None,
        }
    except Exception as e:
        return {"error": f"Could not fetch balance sheet for {ticker}: {str(e)}"}

@tool
def get_cash_flow(ticker: str) -> dict:
    """Get cash flow statement data."""
    try:
        stock = yf.Ticker(ticker.upper())
        cashflow = stock.cashflow
        
        if cashflow.empty:
            return {"error": f"No cash flow data available for {ticker}"}
        
        latest = cashflow.iloc[:, 0]  # Most recent data
        
        return {
            "ticker": ticker.upper(),
            "period": str(cashflow.columns[0].date()),
            "operating_cash_flow": int(latest.get('Operating Cash Flow', 0)) if 'Operating Cash Flow' in latest else None,
            "capital_expenditure": int(latest.get('Capital Expenditure', 0)) if 'Capital Expenditure' in latest else None,
            "free_cash_flow": int(latest.get('Free Cash Flow', 0)) if 'Free Cash Flow' in latest else None,
        }
    except Exception as e:
        return {"error": f"Could not fetch cash flow for {ticker}: {str(e)}"}

# ============= TECHNICAL TOOLS =============
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
def get_historical_data(ticker: str, period: str = "1mo") -> dict:
    """Get historical price data for a stock. Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y."""
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No historical data for {ticker}"}
        
        return {
            "ticker": ticker.upper(),
            "period": period,
            "summary": {
                "high": float(hist['High'].max()),
                "low": float(hist['Low'].min()),
                "avg_volume": int(hist['Volume'].mean()),
                "price_change": float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]),
                "percent_change": float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)
            }
        }
    except Exception as e:
        return {"error": f"Could not fetch history for {ticker}: {str(e)}"}

# ============= SENTIMENT TOOLS =============
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

# ============= SCREENING & DISCOVERY TOOLS =============
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
def screen_stocks_for_growth(tickers_str: str) -> dict:
    """Screen multiple stocks for long-term growth potential. Pass tickers as comma-separated string like 'AAPL,MSFT,GOOGL'."""
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
                score = 50  # Start at neutral
                
                # Strong revenue and earnings growth
                if revenue_growth and revenue_growth > 0.15:
                    score += 15
                if earnings_growth and earnings_growth > 0.15:
                    score += 15
                
                # Good profitability
                if roe and roe > 0.15:
                    score += 10
                if profit_margin and profit_margin > 0.15:
                    score += 10
                
                # Reasonable valuation
                if pe_ratio and 15 < pe_ratio < 30:
                    score += 10
                elif pe_ratio and pe_ratio < 15:
                    score += 5
                elif pe_ratio and pe_ratio > 40:
                    score -= 10
                
                # Good PEG ratio (< 2 is good for growth)
                if peg_ratio and peg_ratio < 1.5:
                    score += 10
                elif peg_ratio and peg_ratio > 3:
                    score -= 5
                
                # Manageable debt
                if debt_to_equity and debt_to_equity < 50:
                    score += 5
                elif debt_to_equity and debt_to_equity > 150:
                    score -= 10
                
                results.append({
                    "ticker": ticker,
                    "company_name": info.get("longName", "N/A"),
                    "growth_score": max(0, min(100, score)),
                    "pe_ratio": pe_ratio,
                    "peg_ratio": peg_ratio,
                    "revenue_growth": f"{revenue_growth*100:.1f}%" if revenue_growth else "N/A",
                    "earnings_growth": f"{earnings_growth*100:.1f}%" if earnings_growth else "N/A",
                    "roe": f"{roe*100:.1f}%" if roe else "N/A",
                    "debt_to_equity": debt_to_equity,
                    "profit_margin": f"{profit_margin*100:.1f}%" if profit_margin else "N/A"
                })
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "error": f"Could not analyze {ticker}"
                })
        
        # Sort by growth score
        results.sort(key=lambda x: x.get("growth_score", 0), reverse=True)
        
        return {
            "screened_stocks": results,
            "top_pick": results[0]["ticker"] if results and results[0].get("growth_score") else None,
            "analysis_note": "Growth score based on revenue growth, earnings growth, ROE, valuation, and debt levels"
        }
    except Exception as e:
        return {"error": f"Screening error: {str(e)}"}

@tool
def compare_stocks(tickers_str: str) -> dict:
    """Compare multiple stocks side by side. Pass tickers as comma-separated string like 'AAPL,MSFT,GOOGL'."""
    try:
        tickers = [t.strip().upper() for t in tickers_str.split(",")]
        comparison = []
        
        for ticker in tickers[:5]:  # Limit to 5 stocks
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                comparison.append({
                    "ticker": ticker,
                    "company": info.get("longName", "N/A"),
                    "price": info.get("currentPrice"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "roe": info.get("returnOnEquity"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "profit_margin": info.get("profitMargins"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "dividend_yield": info.get("dividendYield")
                })
            except:
                comparison.append({"ticker": ticker, "error": "Could not fetch data"})
        
        return {
            "comparison": comparison,
            "stocks_compared": len(comparison)
        }
    except Exception as e:
        return {"error": f"Comparison error: {str(e)}"}

# ============= AGENT STATE ============= 
class AgentState(TypedDict):
    messages: list
    next_agent: str
    agents_consulted: list[str]

# ============= AGENT DEFINITIONS =============
financial_tools = [get_stock_price, get_financial_metrics, get_income_statement, get_balance_sheet, get_cash_flow]
technical_tools = [get_technical_indicators, get_historical_data]
sentiment_tools = [get_news_sentiment]
screening_tools = [get_stock_universe, screen_stocks_for_growth, compare_stocks]

FINANCIAL_AGENT_PROMPT = """You are a Financial Analysis Agent. Your role is to:
- Analyze comprehensive financial metrics (P/E, P/B, ROE, Debt-to-Equity, margins, etc.) for ALL stock(s) mentioned in the conversation
- Assess company valuation and financial health for each stock
- Review income statements, balance sheets, and cash flows
- Evaluate profitability ratios and debt levels
- Compare financial health across multiple stocks when multiple are mentioned

IMPORTANT: Look at the previous messages to identify ALL stock ticker(s) recommended. Analyze EACH stock individually using your tools, then provide a comparison summary.

When you have analyzed all stocks, provide a comprehensive financial analysis comparing all candidates."""

TECHNICAL_AGENT_PROMPT = """You are a Technical Analysis Agent. Your role is to:
- Analyze price trends and patterns for ALL stock(s) mentioned in the conversation
- Evaluate technical indicators (RSI, MACD, EMA, SMA, volume) for each stock
- Identify EMA crossovers (9 and 21 period)
- Assess RSI levels for overbought/oversold conditions
- Analyze MACD signals and histogram
- Identify support/resistance levels
- Compare technical signals across multiple stocks when multiple are mentioned

IMPORTANT: Look at the previous messages to identify ALL stock ticker(s) recommended. Analyze EACH stock individually using your tools, then provide a comparison summary.

When you have analyzed all stocks, respond with your technical analysis comparing all candidates."""

SENTIMENT_AGENT_PROMPT = """You are a Sentiment Analysis Agent. Your role is to:
- Analyze market sentiment and news for ALL stock(s) mentioned in the conversation
- Assess investor sentiment for each stock
- Evaluate news headlines and their impact
- Compare sentiment across multiple stocks when multiple are mentioned

IMPORTANT: Look at the previous messages to identify ALL stock ticker(s) recommended. Analyze EACH stock individually using get_news_sentiment, then provide a comparison summary.

When you have analyzed all stocks, provide your final recommendation on which stock(s) have the best combination of financial health, technical signals, and positive sentiment."""

SCREENING_AGENT_PROMPT = """You are a Stock Screening Agent. Your role is to identify the best stock candidates for investment.

When the user asks for stock recommendations WITHOUT specifying specific tickers (e.g., "what stocks should I invest in", "find good tech stocks", "best stocks this month"), you MUST IMMEDIATELY use your tools:

Step 1: Call get_stock_universe with appropriate category:
- If user mentions 'tech/technology' → use category='tech'
- If user mentions 'finance/bank' → use category='finance'  
- If user mentions 'healthcare/medical' → use category='healthcare'
- If general/no category → use category='large_cap'

Step 2: Call screen_stocks_for_growth with the tickers from step 1 (comma-separated string)

Step 3: Call compare_stocks with the top 2 stocks from step 2

IMPORTANT: To avoid rate limits, recommend only the TOP 2 BEST stocks for detailed analysis.

DO NOT ask the user for specific stock tickers when they're asking for recommendations. USE YOUR TOOLS IMMEDIATELY.

At the end, explicitly state:
"Recommended stocks for detailed analysis: [TICKER1], [TICKER2]"

These stocks will then be analyzed by the Financial, Technical, and Sentiment agents."""

def create_agent(llm, tools, system_prompt):
    """Create an agent with tools."""
    
    def agent(state: AgentState):
        messages = state["messages"]
        
        # Only prepend system instructions to the very first user message
        # Check if we already have AI messages (meaning we're in a later stage)
        has_ai_messages = any(isinstance(msg, AIMessage) for msg in messages)
        
        if not has_ai_messages and messages and isinstance(messages[0], HumanMessage):
            # First agent: prepend system instructions to user query
            first_msg = HumanMessage(content=f"{system_prompt}\n\nUser: {messages[0].content}")
            api_messages = [first_msg]
        else:
            # Subsequent agents: just use the conversation history
            # Add system context as a new user message with clear instructions
            api_messages = messages + [HumanMessage(content=f"{system_prompt}\n\nBased on the conversation above, continue the analysis for the stock ticker(s) that have been mentioned.")]
            
        response = llm.bind_tools(tools).invoke(api_messages)
        return {"messages": [response]}
    
    return agent

def should_continue(state: AgentState):
    """Decide if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# ============= WORKFLOW =============
def create_multi_agent_workflow(api_key: str):
    """Create the multi-agent workflow."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.7,
        max_retries=2
    )
    
    # Create agents
    screening_agent = create_agent(llm, screening_tools, SCREENING_AGENT_PROMPT)
    financial_agent = create_agent(llm, financial_tools, FINANCIAL_AGENT_PROMPT)
    technical_agent = create_agent(llm, technical_tools, TECHNICAL_AGENT_PROMPT)
    sentiment_agent = create_agent(llm, sentiment_tools, SENTIMENT_AGENT_PROMPT)
    
    # Create tool nodes
    screening_tool_node = ToolNode(screening_tools)
    financial_tool_node = ToolNode(financial_tools)
    technical_tool_node = ToolNode(technical_tools)
    sentiment_tool_node = ToolNode(sentiment_tools)
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("screening_agent", screening_agent)
    workflow.add_node("screening_tools", screening_tool_node)
    workflow.add_node("financial_agent", financial_agent)
    workflow.add_node("financial_tools", financial_tool_node)
    workflow.add_node("technical_agent", technical_agent)
    workflow.add_node("technical_tools", technical_tool_node)
    workflow.add_node("sentiment_agent", sentiment_agent)
    workflow.add_node("sentiment_tools", sentiment_tool_node)
    
    # Screening agent flow (first)
    workflow.add_edge(START, "screening_agent")
    workflow.add_conditional_edges("screening_agent", should_continue, {"tools": "screening_tools", END: "financial_agent"})
    workflow.add_edge("screening_tools", "screening_agent")
    
    # Financial agent flow
    workflow.add_conditional_edges("financial_agent", should_continue, {"tools": "financial_tools", END: "technical_agent"})
    workflow.add_edge("financial_tools", "financial_agent")
    
    # Technical agent flow
    workflow.add_conditional_edges("technical_agent", should_continue, {"tools": "technical_tools", END: "sentiment_agent"})
    workflow.add_edge("technical_tools", "technical_agent")
    
    # Sentiment agent flow
    workflow.add_conditional_edges("sentiment_agent", should_continue, {"tools": "sentiment_tools", END: END})
    workflow.add_edge("sentiment_tools", "sentiment_agent")
    
    return workflow.compile()

# ============= API ENDPOINTS =============
@app.post("/analyze", response_model=QueryResponse)
async def analyze_stock(request: QueryRequest):
    """Analyze a stock using multi-agent system."""
    try:
        workflow = create_multi_agent_workflow(request.api_key)
        
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "next_agent": "financial",
            "agents_consulted": []
        }
        
        result = workflow.invoke(initial_state)
        
        # Collect all agent responses
        responses = []
        agents_used = []
        
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                # Handle both string and list content
                if isinstance(msg.content, list):
                    # If content is a list of dicts with 'text' key, extract text
                    text_parts = []
                    for item in msg.content:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                        else:
                            text_parts.append(str(item))
                    content_str = " ".join(text_parts)
                    responses.append(content_str)
                else:
                    responses.append(msg.content)
        
        # Determine which agents were used by checking messages
        message_str = str(result["messages"])
        if "Screening" in message_str or "screen" in message_str.lower():
            agents_used.append("Screening Agent")
        if "Financial" in message_str:
            agents_used.append("Financial Agent")
        if "Technical" in message_str:
            agents_used.append("Technical Agent")
        if "Sentiment" in message_str:
            agents_used.append("Sentiment Agent")
        
        final_response = "\n\n".join(responses)
        
        return QueryResponse(
            response=final_response,
            agents_used=agents_used if agents_used else ["All Agents"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

