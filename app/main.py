from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import yfinance as yf
from datetime import datetime, timezone

app = FastAPI(title="AI Markets – MVP")

# CORS (ώστε να το δεις κι από browser/app)
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

# ------------------- Helpers -------------------
def _ts_to_ms(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)

def _yf_pair(pair: str) -> str:
    """EURUSD -> EURUSD=X (Yahoo Finance)"""
    p = pair.upper().replace("/", "")
    if not (len(p) in (6,7)):  # πρόχειρος έλεγχος
        raise HTTPException(status_code=400, detail="Δώσε π.χ. EURUSD ή GBPUSD")
    if p.endswith("=X"):
        return p
    return f"{p}=X"

# ------------------- Crypto (CoinGecko) -------------------
@app.get("/crypto/price/{symbol}")
async def crypto_price(symbol: str):
    mapping = {
        "btc": "bitcoin", "bitcoin": "bitcoin",
        "eth": "ethereum", "ethereum": "ethereum",
        "sol": "solana", "solana": "solana",
        "ada": "cardano", "cardano": "cardano",
        "xrp": "ripple", "ripple": "ripple",
    }
    coin_id = mapping.get(symbol.lower())
    if not coin_id:
        raise HTTPException(status_code=400, detail="Υποστήριξη: BTC, ETH, SOL, ADA, XRP.")
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="CoinGecko πρόβλημα.")
    price = data.get(coin_id, {}).get("usd")
    if price is None:
        raise HTTPException(status_code=502, detail="Δεν βρέθηκε τιμή.")
    return {"symbol": symbol.upper(), "usd": price}

# ------------------- Stocks / ETFs (Yahoo Finance) -------------------
@app.get("/stock/price/{symbol}")
def stock_price(symbol: str):
    t = yf.Ticker(symbol.upper())
    try:
        fi = t.fast_info
        price = float(fi["last_price"])
        currency = fi.get("currency", "USD")
    except Exception:
        raise HTTPException(status_code=502, detail="Δεν βρέθηκε τιμή (σύμβολο;).")
    return {"symbol": symbol.upper(), "price": price, "currency": currency}

@app.get("/stock/candles/{symbol}")
def stock_candles(symbol: str, interval: str = "1d", range: str = "1mo"):
    """
    interval: 1m, 2m, 5m, 15m, 1h, 1d, 1wk, 1mo
    range: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    try:
        df = yf.download(symbol.upper(), period=range, interval=interval, progress=False)
    except Exception:
        raise HTTPException(status_code=502, detail="Πρόβλημα λήψης δεδομένων.")
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
    return {"symbol": symbol.upper(), "interval": interval, "range": range, "candles": candles}

# ------------------- Forex (Yahoo Finance) -------------------
@app.get("/forex/price/{pair}")
def forex_price(pair: str):
    ysym = _yf_pair(pair)
    t = yf.Ticker(ysym)
    try:
        price = float(t.fast_info["last_price"])
    except Exception:
        raise HTTPException(status_code=502, detail="Δεν βρέθηκε τιμή.")
    return {"pair": pair.upper(), "price": price}

# ------------------- Commodities (Yahoo Finance) -------------------
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
    t = yf.Ticker(sym)
    try:
        price = float(t.fast_info["last_price"])
    except Exception:
        raise HTTPException(status_code=502, detail="Δεν βρέθηκε τιμή.")
    return {"commodity": key, "symbol": sym, "price": price}
