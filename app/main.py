# app/main.py
from __future__ import annotations
import os
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr

# --- logging: δείξε exceptions/traceback στα Render logs ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("ai-markets")

# OpenAI SDK v1
try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError("Απαιτείται το πακέτο 'openai' v1+") from exc

load_dotenv()

app = FastAPI(title="AI Markets Advice API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # προσοχή σε production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AdviceRequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=3)
    risk_profile: Optional[str] = None
    language: Optional[str] = "el"
    model: Optional[str] = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

class AdviceResponse(BaseModel):
    answer: str
    model: str
    disclaimer: str = (
        "Οι πληροφορίες είναι εκπαιδευτικού χαρακτήρα και δεν αποτελούν επενδυτική συμβουλή."
    )

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # γιατί: εμφανές σφάλμα όταν λείπει το key
        logger.error("OPENAI_API_KEY is missing in environment")
        raise HTTPException(status_code=401, detail="Λείπει το OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "You are an investment information assistant. "
    "Always include brief risk summary, time horizon, diversification, fees, and macro risks. "
    "Ask for missing constraints if needed. Bullet-first, then a short paragraph. "
    "Add: 'This is not financial advice.' at the end.\n"
    "Output language: {lang}."
)

@app.get("/")
def root() -> dict:
    return {"ok": True, "service": "ai-markets", "version": "1.0.1"}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}  # Render health check

# --- μικρό debug endpoint: ΔΕΝ δείχνει μυστικά, μόνο αν υπάρχουν ---
@app.get("/debug/env")
def debug_env() -> dict:
    return {
        "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }

@app.post("/advice", response_model=AdviceResponse)
def get_advice(payload: AdviceRequest, client: OpenAI = Depends(get_openai_client)) -> AdviceResponse:
    try:
        logger.info("Advice request received | model=%s risk=%s", payload.model, payload.risk_profile or "unspecified")
        system_msg = SYSTEM_PROMPT.format(lang=payload.language or "el")
        user_msg = _build_user_message(payload)

        completion = client.chat.completions.create(
            model=payload.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=600,
            timeout=30.0,
        )
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            logger.error("Empty response from OpenAI")
            raise HTTPException(status_code=502, detail="Κενή απάντηση από το μοντέλο.")
        return AdviceResponse(answer=content, model=payload.model)

    except HTTPException:
        raise
    except Exception as exc:
        # γιατί: να δούμε ακριβώς το λάθος στο Render logs
        logger.exception("OpenAI call failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"AI error: {exc!s}") from exc

def _build_user_message(p: AdviceRequest) -> str:
    risk = p.risk_profile or "unspecified"
    return (
        f"Client question: {p.question}\n"
        f"Risk profile: {risk}\n"
        "Constraints: prefer diversified, low-cost options; mention risks; "
        "provide 2-3 actionable ideas with tickers where applicable."
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
