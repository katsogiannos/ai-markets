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

app = FastAPI(title="AI Markets", version="1.1.0")

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

def ai_summarize(payload: Dict[str, Any], user_query: Optional[str] = None) -> str:
    if not openai_client:
        return "AI is not configured (missing OPENAIKEY/OPENAI_API_KEY)."

    prompt = (
        "You are an investment research assistant. Summarize and analyze the following "
        "market snapshot in English. Provide:\n"
        "• 3–5 concise bullet points of insights\n"
        "• 2 opportunities and 2 risks\n"
        "• A short balanced outlook (1–2 sentences)\n"
        "DO NOT give financial advice; include a disclaimer. "
        "Use only the data provided; if something is missing, say so briefly.\n\n"
        f"User query (optional): {user_query or '—'}\n\n"
        "Market snapshot:\n" + json.dumps(payload, ensure_ascii=False)
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
# Routes (existing)
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return (
        "<h2>AI Markets is running.</h2>"
        "<p>Try: /healthz, /crypto/price/bitcoin, /stock/price/AAPL, /news, /dashboard, "
        "<a href='/advisor'>/advisor</a></p>"
    )

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

# -----------------------
# NEW: Advisor endpoints
# -----------------------
@app.get("/advisor", response_class=HTMLResponse)
def advisor_page():
    # Simple HTML UI (English) to send a query to /advisor (POST)
    return """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AI Markets Advisor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:820px;margin:32px auto;padding:0 16px}
h1{font-size:1.6rem;margin:0 0 12px}
label{display:block;margin:14px 0 6px}
input,textarea,button{width:100%;box-sizing:border-box;font:inherit}
textarea{min-height:120px}
.row{display:flex;gap:12px}
.row>div{flex:1}
button{margin-top:14px;padding:10px 14px;cursor:pointer}
pre{white-space:pre-wrap;background:#f6f6f6;border:1px solid #e2e2e2;padding:12px;border-radius:8px}
.small{color:#666;font-size:.9rem}
</style>
<h1>AI Markets Advisor</h1>
<p class="small">Type a research question (e.g., “Where should I invest today given high rates?”). This is educational research, not financial advice.</p>

<label>Your question</label>
<textarea id="q" placeholder="Where should I invest today? Consider large-cap tech and BTC."></textarea>

<div class="row">
  <div>
    <label>Stocks/ETFs (comma-separated tickers)</label>
    <input id="tickers" value="AAPL,MSFT,SPY">
  </div>
  <div>
    <label>Coins (comma-separated ids)</label>
    <input id="coins" value="bitcoin,ethereum">
  </div>
</div>

<div class="row">
  <div>
    <label>News limit</label>
    <input id="news" type="number" value="5" min="0" max="20">
  </div>
  <div>
    <label>Language (news)</label>
    <input id="lang" value="en">
  </div>
</div>

<button id="run">Analyze</button>

<h3>Result</h3>
<pre id="out">—</pre>

<script>
const $ = id => document.getElementById(id);
$("run").onclick = async () => {
  $("out").textContent = "Working…";
  const body = {
    query: $("q").value || "",
    tickers: $("tickers").value || "AAPL,MSFT,SPY",
    coins: $("coins").value || "bitcoin,ethereum",
    news_limit: Number($("news").value||5),
    lang: $("lang").value || "en"
  };
  try {
    const r = await fetch("/advisor", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const j = await r.json();
    if (j.analysis) $("out").textContent = j.analysis;
    else $("out").textContent = JSON.stringify(j,null,2);
  } catch(e){
    $("out").textContent = "Error: " + e;
  }
};
</script>
</html>"""

@app.post("/advisor")
def advisor(body: Dict[str, Any] = Body(...)):
    """
    Body:
    {
      "query": "...",
      "coins": "bitcoin,ethereum",
      "tickers": "AAPL,MSFT,SPY",
      "news_limit": 5,
      "lang": "en"
    }
    """
    if not openai_client:
        raise HTTPException(400, "AI is not configured (set OPENAIKEY / OPENAI_API_KEY).")

    query = (body.get("query") or "").strip()
    coins = (body.get("coins") or "bitcoin,ethereum").strip()
    tickers = (body.get("tickers") or "AAPL,MSFT,SPY").strip()
    news_limit = int(body.get("news_limit") or 5)
    lang = (body.get("lang") or "en").strip()

    # Gather snapshot (reuse logic from /dashboard)
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

    analysis = ai_summarize(snapshot, user_query=query)
    return {"analysis": analysis, "data": snapshot}
