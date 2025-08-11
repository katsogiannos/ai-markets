from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx, yfinance as yf
from datetime import datetime, timezone

app = FastAPI(title="AI Markets – MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    return "It works ✅"

# ---------- Helpers ----------
def _ts_to_ms(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)

def yf_last_price(symbol: str):
    """Πάρε τιμή με 2 προσπάθειες: fast_info -> history."""
    t = yf.Ticker(symbol)
    # 1) fast_info
    try:
        fi = t.fast_info
        p = fi.get("last_price")
        if p is not None:
            return float(p), fi.get("currency", "USD")
    except Exception:
        pass
    # 2) τελευταία τιμή από history
    try:
        h = t.history(period="1d", interval="1d", auto_adjust=False, actions=False)
        if h is not None and not h.empty:
            return float(h["Close"][-1]), "USD"
    except Exception:
        pass
    raise HTTPException(status_code=502, detail="Δεν βρέθηκε τιμή.")

# ---------- Crypto (CoinGecko) ----------
@app.get("/crypto/price/{symbol}")
async def crypto_price(symbol: str):
    mapping = {
        "btc": "bitcoin", "bitcoin": "bitcoin",
        "eth": "ethereum", "ethereum": "ethereum",
        "sol": "solana", "solana": "solana",
        "ada": "cardano", "cardano": "cardano",
        "xrp": "ripple",  "ripple": "ripple",
    }
    coin_id = mapping.get(symbol.lower())
    if not coin_id:
        raise HTTPException(status_code=400, detail="Υποστήριξη: BTC, ETH, SOL, ADA, XRP.")
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    headers = {
        # Κάποιοι providers θέλουν User-Agent
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="CoinGecko πρόβλημα.")
    price = data.get(coin_id, {}).get("usd")
    if price is None:
        raise HTTPException(status_code=502, detail="Δεν βρέθηκε τιμή.")
    return {"symbol": symbol.upper(), "usd": price}

# ---------- Stocks / ETFs ----------
@app.get("/stock/price/{symbol}")
def stock_price(symbol: str):
    price, ccy = yf_last_price(symbol.upper())
    return {"symbol": symbol.upper(), "price": price, "currency": ccy}

@app.get("/stock/candles/{symbol}")
def stock_candles(symbol: str, interval: str = "1d", range: str = "1mo"):
    """
    interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,1wk,1mo
    range: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    sym = symbol.upper()
    # δοκίμασε history (πιο αξιόπιστο στο Render)
    try:
        df = yf.Ticker(sym).history(period=range, interval=interval, auto_adjust=False, actions=False)
    except Exception:
        df = None
    if df is None or df.empty:
        # δεύτερη προσπάθεια με download
        try:
            df = yf.download(sym, period=range, interval=interval, progress=False, threads=False, prepost=False)
        except Exception:
            df = None
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Κενό αποτέλεσμα.")
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "t": _ts_to_ms(ts.to_pydatetime()),
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
            "v": float(row.get("Volume", 0) or 0),
        })
    return {"symbol": sym, "interval": interval, "range": range, "candles": candles}

# ---------- Forex ----------
@app.get("/forex/price/{pair}")
def forex_price(pair: str):
    p = pair.upper().replace("/", "")
    ysym = f"{p}=X"
    price, _ = yf_last_price(ysym)
    return {"pair": pair.upper(), "price": price}

# ---------- Commodities ----------
@app.get("/commodity/price/{name}")
def commodity_price(name: str):
    m = {
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "WTI": "CL=F",
        "BRENT": "BZ=F",
        "NATGAS": "NG=F",
    }
    key = name.upper()
    sym = m.get(key)
    if not sym:
        raise HTTPException(status_code=400, detail="Υποστήριξη: GOLD, SILVER, WTI, BRENT, NATGAS.")
    price, _ = yf_last_price(sym)
    return {"commodity": key, "symbol": sym, "price": price}
