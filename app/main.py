from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
import os
from typing import Any, Dict, List, Optional
import httpx
import csv
from io import StringIO

# -----------------------------
# App & Config (MUST be first)
# -----------------------------
app = FastAPI(title="AI Markets")

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAIKEY")
    or os.getenv("OPEN_AI_KEY")
)

# OpenAI (new SDK)
try:
    from openai import OpenAI  # pip install openai>=1.40
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None


# -----------------------------
# Helpers
# -----------------------------
HTTP_TIMEOUT = 12.0


async def fetch_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def fetch_text(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.text


def ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data}


def fail(msg: str, status: int = 502):
    raise HTTPException(status, msg)


# -----------------------------
# Basic endpoints
# -----------------------------
@app.get("/", response_class=PlainTextResponse)
def home():
    return (
        "AI Markets is running.\n\n"
        "Try:\n"
        "  /healthz\n"
        "  /crypto/price/bitcoin\n"
        "  /stock/price/AAPL\n"
        "  /forex/price/EURUSD\n"
        "  /advisor (simple UI)\n"
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


# -----------------------------
# Data endpoints (free providers)
# -----------------------------
@app.get("/crypto/price/{coin_id}")
async def crypto_price(coin_id: str):
    """
    Free source: CoinGecko Simple Price (no key required).
    Example: /crypto/price/bitcoin
    """
    try:
        data = await fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id.lower(), "vs_currencies": "usd"},
        )
        if coin_id.lower() not in data:
            return ok({"id": coin_id, "usd": None, "note": "coin not found"})
        return ok({"id": coin_id, "usd": data[coin_id.lower()]["usd"]})
    except Exception as e:
        fail(f"coingecko error: {e}")


@app.get("/stock/price/{ticker}")
async def stock_price(ticker: str):
    """
    Free source: Stooq (CSV). We try {TICKER}.US e.g., AAPL.US
    Example: /stock/price/AAPL
    """
    sym = f"{ticker.upper()}.US"
    url = "https://stooq.com/q/l/"
    params = {"s": sym, "i": "d"}
    try:
        txt = await fetch_text(url, params=params)
        # CSV like: Symbol,Date,Time,Open,High,Low,Close,Volume
        # AAPL.US,2024-08-09,22:00:09,XXXX,XXXX,XXXX,189.97,xxxxx
        f = StringIO(txt)
        rows = list(csv.reader(f))
        if len(rows) >= 2 and rows[1][0].upper() == sym.upper():
            close_str = rows[1][6]
            price = float(close_str) if close_str not in ("N/D", "N/A", "") else None
            return ok({"ticker": ticker.upper(), "price": price, "source": "stooq"})
        return ok({"ticker": ticker.upper(), "price": None, "note": "not found"})
    except Exception as e:
        fail(f"stooq error: {e}")


@app.get("/forex/price/{pair}")
async def forex_price(pair: str):
    """
    Free source: exchangerate.host convert endpoint.
    Example: /forex/price/EURUSD (FROM=EUR, TO=USD)
    """
    pair = pair.upper().strip().replace("/", "")
    if len(pair) != 6:
        fail("pair must be like EURUSD", 400)
    base, quote = pair[:3], pair[3:]
    try:
        data = await fetch_json(
            "https://api.exchangerate.host/convert",
            params={"from": base, "to": quote},
        )
        return ok({"pair": pair, "rate": data.get("result")})
    except Exception as e:
        fail(f"fx error: {e}")


# -----------------------------
# AI endpoints (chat + advisor UI)
# -----------------------------
SYSTEM_PROMPT = (
    "You are an investment research assistant. Be helpful, concise, and neutral. "
    "You can discuss crypto, stocks, ETFs, forex and macro news. "
    "Always include a disclaimer that this is not financial advice."
)


@app.post("/chat")
async def chat(payload: Dict[str, Any] = Body(...)):
    """
    Generic chat endpoint that can accept:
    {
      "question": "text",           (free-form question)
      "tickers": "AAPL,MSFT,SPY",   (optional)
      "coins": "bitcoin,ethereum",  (optional)
      "news_limit": 5,              (ignored for now)
      "lang": "en"
    }
    It fetches some live prices and passes context to the LLM.
    """
    if not openai_client:
        fail("AI is not configured (set OPENAI_API_KEY).", 400)

    question: str = (payload.get("question") or "").strip()
    tickers_raw: str = (payload.get("tickers") or "").strip()
    coins_raw: str = (payload.get("coins") or "").strip()
    lang: str = (payload.get("lang") or "en").strip()

    if not question:
        fail("question is required", 400)

    tickers: List[str] = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    coins: List[str] = [c.strip().lower() for c in coins_raw.split(",") if c.strip()]

    # gather minimal live context
    context_lines: List[str] = []

    # stocks
    for t in tickers[:8]:
        try:
            sp = await stock_price(t)
            context_lines.append(f"Stock {t} price (stooq): {sp['data']['price']}")
        except Exception:
            context_lines.append(f"Stock {t} price: unavailable")

    # crypto
    for c in coins[:8]:
        try:
            cp = await crypto_price(c)
            context_lines.append(f"Crypto {c} price (usd): {cp['data']['usd']}")
        except Exception:
            context_lines.append(f"Crypto {c} price: unavailable")

    context = "\n".join(context_lines) if context_lines else "No live prices fetched."

    user_prompt = (
        f"Language: {lang}\n"
        f"User question: {question}\n\n"
        f"Available live data:\n{context}\n\n"
        "Please provide a short, structured research answer with:\n"
        "- Key points\n- Pros & cons\n- Simple next steps or what to watch\n"
        "Finish with a disclaimer that this is educational research, not financial advice."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # OpenAI "Responses" API
        resp = openai_client.responses.create(
            model="gpt-5-mini",
            input=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        reply = getattr(resp, "output_text", "").strip() or "No AI output."
        return {"ok": True, "reply": reply}
    except Exception as e:
        fail(f"AI error: {e}")


# -----------------------------
# Simple advisor UI page
# -----------------------------
ADVISOR_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AI Markets Advisor</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; max-width: 900px; margin: 0 auto;}
  input, textarea { width: 100%; padding: 10px; margin: 6px 0 12px; font-size: 15px; box-sizing: border-box; }
  button { padding: 10px 16px; font-size: 15px; cursor: pointer; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  pre { background: #f6f7f9; padding: 14px; border-radius: 8px; white-space: pre-wrap;}
</style>
<h2>AI Markets Advisor</h2>
<p>Ask in free form (e.g., "Where should I invest today given high rates?"). This is educational research, not financial advice.</p>

<label>Your question</label>
<textarea id="q" rows="4" placeholder="e.g., Where should I invest today?"></textarea>

<div class="row">
  <div>
    <label>Stocks/ETFs (comma separated tickers)</label>
    <input id="tickers" placeholder="AAPL,MSFT,SPY" />
  </div>
  <div>
    <label>Coins (comma separated ids)</label>
    <input id="coins" placeholder="bitcoin,ethereum" />
  </div>
</div>

<div class="row">
  <div>
    <label>News limit (not used yet)</label>
    <input id="news" value="5" />
  </div>
  <div>
    <label>Language</label>
    <input id="lang" value="en" />
  </div>
</div>

<button id="go">Analyze</button>

<h3>Result</h3>
<pre id="out"></pre>

<script>
async function callChat() {
  const out = document.getElementById('out');
  out.textContent = "Working...";
  const payload = {
    question: document.getElementById('q').value,
    tickers: document.getElementById('tickers').value,
    coins: document.getElementById('coins').value,
    news_limit: Number(document.getElementById('news').value || 5),
    lang: document.getElementById('lang').value || 'en'
  };
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (!j.ok) {
      out.textContent = "Error: " + JSON.stringify(j);
    } else {
      out.textContent = j.reply || JSON.stringify(j);
    }
  } catch (e) {
    out.textContent = "AI error: " + e;
  }
}
document.getElementById('go').addEventListener('click', callChat);
</script>
</html>
"""

@app.get("/advisor", response_class=HTMLResponse)
def advisor_page():
    return ADVISOR_HTML

