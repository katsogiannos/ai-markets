from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import httpx

app = FastAPI()

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    return "It works ✅"

# /crypto/price/BTC  ή  /crypto/price/ETH  /SOL  /ADA
@app.get("/crypto/price/{symbol}")
async def crypto_price(symbol: str):
    mapping = {
        "btc": "bitcoin", "bitcoin": "bitcoin",
        "eth": "ethereum", "ethereum": "ethereum",
        "sol": "solana", "sol": "solana",
        "ada": "cardano", "cardano": "cardano",
    }
    coin_id = mapping.get(symbol.lower())
    if not coin_id:
        raise HTTPException(status_code=400, detail="Υποστήριξη: BTC, ETH, SOL, ADA.")
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Πρόβλημα με τον πάροχο τιμών.")
    price = data.get(coin_id, {}).get("usd")
    if price is None:
        raise HTTPException(status_code=502, detail="Δεν βρέθηκε τιμή.")
    return {"symbol": symbol.upper(), "usd": price}
