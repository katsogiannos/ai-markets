import os
import json
from typing import List, Optional

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ==== FastAPI App ====
app = FastAPI(title="AI Markets")

# Static / Templates
# (δημιούργησε φάκελο app/templates με το advisor.html από κάτω)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ==== Utilities ====

def get_env_openai_key() -> Optional[str]:
    """
    Παίρνει το OpenAI API key από διάφορα πιθανά ονόματα env var
    γιατί τα είδαμε σε διαφορετικές μορφές στα screenshots.
    """
    candidates = [
        "OPENAI_KEY",
        "OPEN_AI_KEY",
        "OPENAI_API_KEY",
        "OPENAIKEY",
        "OPENAI"
    ]
    for name in candidates:
        val = os.getenv(name)
        if val and val.strip():
            return val.strip()
    return None


def fetch_stock_price_yahoo(symbol: str) -> Optional[float]:
    """
    Προσπαθεί να φέρει τιμή με ένα απλό endpoint του Yahoo (μη επίσημο).
    Αν δεν δουλέψει, επιστρέφει None (θα χειριστούμε graceful fail).
    """
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        quote = data.get("quoteResponse", {}).get("result", [])
        if not quote:
            return None
        price = quote[0].get("regularMarketPrice")
        return float(price) if price is not None else None
    except Exception:
        return None


def fetch_crypto_price_coingecko(ids: List[str], vs="usd") -> dict:
    """
    Δωρεάν CoinGecko simple price (χωρίς κλειδί).
    ids: π.χ. ["bitcoin","ethereum"]
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": ",".join(ids), "vs_currencies": vs}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_fx_price_exchangerate(base: str, quote: str) -> Optional[float]:
    """
    Δωρεάν exchangerate.host
    """
    try:
        url = f"https://api.exchangerate.host/latest?base={base.upper()}&symbols={quote.upper()}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        rate = data.get("rates", {}).get(quote.upper())
        return float(rate) if rate is not None else None
    except Exception:
        return None


def fetch_news_yahoo(tickers: List[str], limit: int = 5, lang: str = "en") -> List[dict]:
    """
    Πολύ απλό aggregator: τραβάει Yahoo Finance RSS ανά ticker.
    Χωρίς κλειδί. Επιστρέφει λίστα άρθρων (τίτλος, link).
    """
    import xml.etree.ElementTree as ET

    items = []
    for symbol in tickers:
        try:
            rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang={lang}"
            r = requests.get(rss, timeout=10)
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                if title and link:
                    items.append({"symbol": symbol.upper(), "title": title, "link": link})
        except Exception:
            continue
        if len(items) >= limit:
            break
    return items[:limit]


# ==== Routes ====

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Landing με links."""
    html = """
    <h3>AI Markets is running.</h3>
    <p>Try: <a href="/healthz">/healthz</a>, 
    <a href="/stock/price/AAPL">/stock/price/AAPL</a>, 
    <a href="/crypto/price/bitcoin">/crypto/price/bitcoin</a>, 
    <a href="/forex/price?base=USD&quote=EUR">/forex/price?base=USD&quote=EUR</a>, 
    <a href="/news?tickers=AAPL,MSFT&limit=5&lang=en">/news</a>,
    <a href="/advisor">/advisor</a></p>
    """
    return HTMLResponse(html)


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---- Stocks ----
@app.get("/stock/price/{symbol}")
def stock_price(symbol: str):
    price = fetch_stock_price_yahoo(symbol)
    if price is None:
        return JSONResponse({"detail": "No price found."}, status_code=404)
    return {"symbol": symbol.upper(), "price": price}


# ---- Crypto ----
@app.get("/crypto/price/{coin_id}")
def crypto_price(coin_id: str, vs: str = "usd"):
    data = fetch_crypto_price_coingecko([coin_id], vs=vs)
    if not data or coin_id not in data or vs not in data[coin_id]:
        return JSONResponse({"detail": "CoinGecko error."}, status_code=502)
    return {"coin": coin_id, "vs": vs, "price": data[coin_id][vs]}


# ---- Forex ----
@app.get("/forex/price")
def fx_price(base: str = "USD", quote: str = "EUR"):
    rate = fetch_fx_price_exchangerate(base, quote)
    if rate is None:
        return JSONResponse({"detail": "No FX rate found."}, status_code=404)
    return {"base": base.upper(), "quote": quote.upper(), "rate": rate}


# ---- News ----
@app.get("/news")
def news(tickers: str = "AAPL", limit: int = 5, lang: str = "en"):
    tick_list = [t.strip() for t in tickers.split(",") if t.strip()]
    items = fetch_news_yahoo(tick_list, limit=limit, lang=lang)
    return {"count": len(items), "items": items}


# ---- Advisor page (UI) ----
@app.get("/advisor", response_class=HTMLResponse)
def advisor_page(request: Request):
    return templates.TemplateResponse("advisor.html", {"request": request})


# ---- Advisor API (OpenAI) ----
@app.post("/api/advice")
async def api_advice(payload: dict):
    """
    payload: {
      "question": str,
      "stocks": "AAPL,MSFT",
      "coins": "bitcoin,ethereum",
      "newsLimit": 5,
      "newsLang": "en"
    }
    """
    question = (payload.get("question") or "").strip()
    stocks = [s.strip() for s in (payload.get("stocks") or "").split(",") if s.strip()]
    coins  = [c.strip() for c in (payload.get("coins") or "").split(",") if c.strip()]
    news_limit = int(payload.get("newsLimit") or 5)
    news_lang  = (payload.get("newsLang") or "en").strip() or "en"

    # Μαζεύουμε δεδομένα
    stock_data = []
    for s in stocks:
        p = fetch_stock_price_yahoo(s)
        if p is not None:
            stock_data.append({"symbol": s.upper(), "price": p})

    crypto_data = {}
    if coins:
        cg = fetch_crypto_price_coingecko(coins, vs="usd")
        for cid in coins:
            val = cg.get(cid, {}).get("usd")
            if val is not None:
                crypto_data[cid] = val

    headlines = fetch_news_yahoo(stocks or ["AAPL"], limit=news_limit, lang=news_lang)

    # Prompt για OpenAI
    openai_key = get_env_openai_key()
    if not openai_key:
        # Αν δεν έχουμε κλειδί, επιστρέφουμε μια "ανθρώπινη" σύνοψη
        return {
            "ai_summary": "OpenAI key not configured. Here is a simple summary.",
            "stocks": stock_data,
            "crypto": crypto_data,
            "news": headlines,
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        context = {
            "question": question,
            "stocks": stock_data,
            "crypto": crypto_data,
            "news": headlines,
            "disclaimer": "Educational research only. Not financial advice."
        }

        # Μικρό, οικονομικό μοντέλο για περίληψη
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful markets research assistant. Provide clear, practical, balanced insights. Never give financial advice; include a short disclaimer."},
                {"role": "user", "content": f"Summarize the following context and answer the user's question. Context JSON:\n{json.dumps(context, ensure_ascii=False)}"}
            ],
            temperature=0.3
        )
        summary = completion.choices[0].message.content.strip() if completion and completion.choices else "No summary."

        return {
            "ai_summary": summary,
            "stocks": stock_data,
            "crypto": crypto_data,
            "news": headlines,
        }
    except Exception as e:
        return JSONResponse({"detail": f"AI error: {str(e)}"}, status_code=502)

