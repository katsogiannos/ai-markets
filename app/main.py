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
