"""
FastAPI Backend — XAI Spam Classifier
======================================
• Loads DistilBERT from ./model/ (local) if present, else from HuggingFace hub.
• Exposes  GET  /health
           POST /predict   — fast label + probabilities
           POST /explain   — label + probabilities + per-token SHAP values
"""

import os
import shap
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(app.py))
LOCAL_MODEL_DIR = os.path.join(_HERE, "model")
HF_MODEL_NAME = "mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection"

# Pick local cache if already downloaded, otherwise use HF hub
MODEL_SOURCE = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HF_MODEL_NAME

# ──────────────────────────────────────────────────────────────
# Global state (loaded once at startup)
# ──────────────────────────────────────────────────────────────
_classifier = None
_explainer = None
_tokenizer = None


def _load_model():
    global _classifier, _explainer, _tokenizer
    print(f"[backend] Loading model from: {MODEL_SOURCE}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
    model.eval()

    _classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=_tokenizer,
        return_all_scores=True,
        device=-1,          # CPU only
        truncation=True,
        max_length=512,
    )

    # SHAP PartitionExplainer — word-level attribution
    _explainer = shap.Explainer(
        _classifier,
        masker=_tokenizer,
        output_names=["Ham", "Spam"],
    )
    print("[backend] Model + SHAP explainer ready.")


# ──────────────────────────────────────────────────────────────
# Lifespan (modern FastAPI startup/shutdown)
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    # shutdown — nothing to clean up for CPU inference


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="XAI Spam Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class ExplanationToken(BaseModel):
    token: str
    shap_value: float


class ExplanationResponse(BaseModel):
    text: str
    label: str
    spam_probability: float
    ham_probability: float
    base_value: float
    tokens: list[ExplanationToken]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _parse_scores(raw) -> tuple[float, float, str]:
    """
    Robustly extract ham/spam probabilities regardless of whether
    the model returns LABEL_0/LABEL_1, HAM/SPAM, or ham/spam.

    Handles two pipeline output shapes:
      • list[dict]  — [{'label': 'LABEL_0', 'score': 0.9}, ...]
      • list[list]  — [[{'label': 'LABEL_0', 'score': 0.9}, ...]]
        (batch mode wraps results in an extra list)
    """
    # Unwrap an extra batch dimension if present
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    if isinstance(raw, dict):
        raw = [raw]
    # is malformed — raise early with a clear message.
    if not raw or not isinstance(raw[0], dict):
        raise ValueError(
            f"Unexpected classifier output format: {raw!r}. "
            "Expected list[dict] with 'label' and 'score' keys."
        )

    scores = {r["label"].upper(): r["score"] for r in raw}
    # LABEL_0 → Ham,  LABEL_1 → Spam  (most fine-tuned DistilBERT models)
    ham = scores.get("LABEL_0", scores.get("HAM", 0.0))
    spam = scores.get("LABEL_1", scores.get("SPAM", 0.0))
    label = "Spam" if spam >= 0.5 else "Ham (Not Spam)"
    return float(ham), float(spam), label


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_source": MODEL_SOURCE,
        "model_loaded": _classifier is not None,
    }


@app.post("/predict")
async def predict(req: PredictionRequest):
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    ham, spam, label = _parse_scores(_classifier([req.text]))
    return {
        "text": req.text,
        "label": label,
        "spam_probability": round(spam, 5),
        "ham_probability": round(ham, 5),
    }


@app.post("/explain", response_model=ExplanationResponse)
async def explain(req: PredictionRequest):
    if _classifier is None or _explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Prediction
    ham, spam, label = _parse_scores(_classifier([req.text])[0])

    # SHAP word-level attribution
    shap_values = _explainer([req.text])
    # shap_values.values: shape (1, n_tokens, 2)  → index 1 = Spam class
    spam_shap = shap_values.values[0][:, 1]
    tokens = shap_values.data[0]
    base_value = float(shap_values.base_values[0][1])

    token_list = [
        ExplanationToken(token=str(t), shap_value=round(float(v), 6))
        for t, v in zip(tokens, spam_shap)
        if str(t).strip()
    ]

    return ExplanationResponse(
        text=req.text,
        label=label,
        spam_probability=round(spam, 5),
        ham_probability=round(ham, 5),
        base_value=round(base_value, 6),
        tokens=token_list,
    )


# ──────────────────────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import socket
    
    # This block finds a port that is GUARANTEED to be empty
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    free_port = sock.getsockname()[1]
    sock.close()
    
    print(f"--- SIMULATION BOUND TO PORT: {free_port} ---")
    
    # We use 'app' directly instead of the "backend:app" string 
    # to ensure it uses the code currently in THIS file.
    uvicorn.run(app, host="127.0.0.1", port=free_port, log_level="info")