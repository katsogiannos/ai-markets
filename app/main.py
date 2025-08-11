import os
import json
from typing import List, Optional, Dict, Any
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Load .env only for local runs (Render διαβάζει από env vars)
load_dotenv()

# ---- Keys (υποστήριξη και των δύο ονομάτων) ----
OPENAI_API_KEY = os.getenv("OPENAIKEY") or os.getenv("OPENAI_API_KEY")
NEWS_API_KEY   = os.getenv("NEWSAPIKEY") or os.getenv("NEWS_API_KEY")

# ---- OpenAI client (new SDK) ----
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

app = FastAPI(title="AI Markets", version="1.0.0")

# -----------------------
# Helpers
# -----------------------
def coingecko_price(ids: List[str], vs: str = "usd") -> Dict[str, Any]:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": vs}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise HTTPException(502, f"CoinGecko error: {r.text}")
    return r.json()

def yahoo_last_close(ticker: str) -> Optional[float]:
    data = yf.Ticker(ticker).history(period="1d")
    if data is None or data.empty:
        return None
    return float(data["Close"].iloc[-1])

def newsapi_top_business(limit: int = 10, lang: str = "en") -> List[Dict[str, Any]]:
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": "business", "language": lang, "pageSize": limit, "apiKey": NEWS_API_KEY}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return []
    return (r.json().get("articles") or [])[:limit]

def ai_summarize(payload: Dict[str, Any]) -> str:
    if not openai_client:
        return "AI is not configured (missing OPENAIKEY/OPENAI_API_KEY)."
    prompt = (
        "Summarize concisely the following market snapshot in English. "
        "Give 3–5 bullet points and one short outlook sentence. "
        "Data:\n" + json.dumps(payload, ensure_ascii=False)
    )
    try:
        resp = openai_client.responses.create(
            model="gpt-5-mini",
            input=prompt,
        )
        return resp.output_text.strip() if hasattr(resp, "output_text") else "No AI output."
    except Exception as e:
        return f"AI error: {e}"

# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>AI Markets is running.</h2><p>Try: /healthz, /crypto/price/bitcoin, /stock/price/AAPL, /news, /dashboard</p>"

@app.get("/healthz")
def healthz():
    return {"ok": True}

# Crypto
@app.get("/crypto/price/{coin_id}")
def crypto_price(coin_id: str, vs: str = "usd"):
    data = coingecko_price([coin_id], vs)
    if coin_id not in data:
        raise HTTPException(404, f"No price for '{coin_id}'.")
    return {coin_id: data[coin_id]}

@app.get("/crypto/prices")
def crypto_prices(ids: str = "bitcoin,ethereum,solana", vs: str = "usd"):
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    return coingecko_price(id_list, vs)

# Stocks / ETFs
@app.get("/stock/price/{ticker}")
def stock_price(ticker: str):
    price = yahoo_last_close(ticker)
    if price is None:
        raise HTTPException(404, f"No price for '{ticker}'.")
    return {"ticker": ticker.upper(), "close": price}

# News
@app.get("/news")
def top_news(limit: int = 10, lang: str = "en"):
    articles = newsapi_top_business(limit=limit, lang=lang)
    slim = [
        {
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
        }
        for a in articles
    ]
    return {"count": len(slim), "articles": slim}

# AI summary for arbitrary payload
@app.post("/ai/summary")
def ai_summary(body: Dict[str, Any] = Body(...)):
    text = ai_summarize(body)
    return {"summary": text}

# Combined dashboard
@app.get("/dashboard")
def dashboard(
    coins: str = "bitcoin,ethereum,solana",
    tickers: str = "AAPL,MSFT,SPY",
    news_limit: int = 5,
    lang: str = "en",
):
    # Crypto
    coin_ids = [x.strip() for x in coins.split(",") if x.strip()]
    try:
        crypto = coingecko_price(coin_ids, "usd")
    except HTTPException:
        crypto = {}

    # Stocks
    result_stocks = {}
    for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        try:
            p = yahoo_last_close(t)
            if p is not None:
                result_stocks[t] = p
        except Exception:
            pass

    # News
    articles = newsapi_top_business(limit=news_limit, lang=lang)

    snapshot = {
        "crypto": crypto,
        "stocks": result_stocks,
        "news": [
            {"title": a.get("title"), "source": (a.get("source") or {}).get("name")}
            for a in articles
        ],
    }

    summary = ai_summarize(snapshot)
    return {"data": snapshot, "ai_summary": summary}

