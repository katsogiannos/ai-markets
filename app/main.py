# app/main.py
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr

# OpenAI SDK v1
try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError("Το OpenAI SDK v1 (package 'openai') δεν είναι εγκατεστημένο ή έχει παλιά έκδοση.") from exc

load_dotenv()  # φορτώνει OPENAI_API_KEY από .env (αν υπάρχει)

app = FastAPI(title="AI Markets Advice API", version="1.0.0")

# CORS: άλλαξε τα origins ανάλογα με το front-end σου
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # προσοχή για production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AdviceRequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=3) = Field(
        ..., description="Η ερώτηση του πελάτη για αγορές/επενδύσεις."
    )
    risk_profile: Optional[str] = Field(
        default=None,
        description="Προαιρετικό προφίλ ρίσκου (π.χ. conservative | balanced | aggressive).",
    )
    language: Optional[str] = Field(
        default="el",
        description="Γλώσσα απάντησης (el/en).",
    )
    model: Optional[str] = Field(
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        description="Το OpenAI chat model.",
    )


class AdviceResponse(BaseModel):
    answer: str
    model: str
    disclaimer: str = (
        "Οι παρακάτω πληροφορίες είναι εκπαιδευτικού χαρακτήρα και δεν αποτελούν "
        "χρηματοοικονομική ή επενδυτική συμβουλή."
    )


def get_openai_client() -> OpenAI:
    """Φτιάχνει client από το env. Αν λείπει το key, ρίχνει 401."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # γιατί: καλύτερο μήνυμα για σβηστό/λάθος key
        raise HTTPException(status_code=401, detail="Λείπει το OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


SYSTEM_PROMPT = (
    "You are an investment information assistant. "
    "Always include a brief risk summary, time horizon considerations, "
    "diversification, fees/ETFs vs single stocks, and macro risks. "
    "Ask for missing constraints if needed. "
    "KEEP answers concise, bullet-first, then a short paragraph. "
    "Add: 'This is not financial advice.' at the end.\n"
    "Output language: {lang}."
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/advice", response_model=AdviceResponse)
def get_advice(
    payload: AdviceRequest,
    client: OpenAI = Depends(get_openai_client),
) -> AdviceResponse:
    """Καλεί OpenAI Chat Completions API με σωστό schema για SDK v1."""
    try:
        # γιατί: κρατάμε system prompt σταθερό και μικρό context window
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
            timeout=30.0,  # httpx timeout (μέσω SDK)
        )

        content = (completion.choices[0].message.content or "").strip()
        if not content:
            raise HTTPException(status_code=502, detail="Κενή απάντηση από το μοντέλο.")
        return AdviceResponse(answer=content, model=payload.model)
    except HTTPException:
        raise
    except Exception as exc:
        # γιατί: ενιαίος, καθαρός χειρισμός σφαλμάτων
        raise HTTPException(status_code=500, detail=f"AI error: {exc!s}") from exc


def _build_user_message(p: AdviceRequest) -> str:
    """Συνθέτει καθαρά το user prompt."""
    risk = p.risk_profile or "unspecified"
    return (
        f"Client question: {p.question}\n"
        f"Risk profile: {risk}\n"
        "Constraints: prefer diversified, low-cost options; mention risks; "
        "provide 2-3 actionable ideas with tickers where applicable."
    )


if __name__ == "__main__":
    import uvicorn

    # γιατί: εκκίνηση απευθείας για τοπικές δοκιμές
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

