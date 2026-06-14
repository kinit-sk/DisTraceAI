"""Check-worthiness detection.

Single PyTorch/HuggingFace backend for both xlm-multicw and mdb-multicw.
Mirrors HFTextClassifier.load_model() + evaluate() from full-evaluation.py:
  - AutoTokenizer.from_pretrained(model_path)
  - AutoModelForSequenceClassification.from_pretrained(model_path)
  - DataLoader with max_length=256, padding="max_length", argmax decoding

Both xlm-multicw and mdb-multicw are standard HuggingFace checkpoint
directories. The canonical paths Models/xlm-multicw and Models/mdb-multicw
are written by full-evaluation.py after it selects the best seed.
Model path resolution tries both 'Models/' (capital M, training script) and
'models/' (lowercase) so either layout works.
"""
from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal dataset — inference only (no labels tensor)
# ---------------------------------------------------------------------------

class _TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int = 256) -> None:
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

def _resolve_path(model_path: str) -> str | None:
    """Return the first existing path among several candidates.

    Tries the path as given first, then looks under both 'Models/' (capital M
    — where the training script saves checkpoints) and 'models/' (lowercase).
    """
    if not model_path:
        return None
    name = os.path.basename(model_path.rstrip("/\\"))
    candidates = [
        model_path,
        os.path.join("Models", name),   # capital M — matches training script
        os.path.join("models", name),   # lowercase fallback
        os.path.join("Models", model_path),
        os.path.join("models", model_path),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CheckWorthinessDetector:
    """Wraps a fine-tuned HuggingFace sequence-classification checkpoint.

    Works for both xlm-multicw (XLM-RoBERTa) and mdb-multicw (mDeBERTa-v3):
    both were trained with the same HFTextClassifier code and saved as
    standard HuggingFace checkpoint directories.
    """

    def __init__(self, model_path: str, device: str | None = None,
                 max_len: int = 256, batch_size: int = 32) -> None:
        resolved = _resolve_path(model_path)
        if resolved is None:
            raise FileNotFoundError(
                f"CW model not found at {model_path!r}. "
                f"Tried Models/<name> and models/<name>. "
                f"Place the checkpoint directory under Models/.")

        if os.environ.get("DISTRACE_CW_CPU") == "1":
            device = "cpu"
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len    = max_len
        self.batch_size = batch_size
        self.model_path = resolved

        # Mirror load_model() from full-evaluation.py:
        # both tokenizer and model load from the checkpoint directory, no
        # extra flags — the saved tokenizer_config.json picks the right class.
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True)

        # Prefer safetensors when loading — safetensors is exempt from the
        # torch.load CVE-2025-32434 restriction (requires torch>=2.6 for .bin).
        # If no model.safetensors exists in the checkpoint the flag is silently
        # ignored and HuggingFace falls back to pytorch_model.bin (which then
        # requires torch>=2.6 in newer transformers versions).
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved, use_safetensors=True).to(self.device)
        self.model.eval()
        logger.info("[cw] loaded %s on %s", resolved, self.device)

    # ------------------------------------------------------------------ #

    def predict(self, sentences: list[str]) -> list[int]:
        """Return a 0/1 label per sentence (1 = check-worthy)."""
        if not sentences:
            return []
        loader = DataLoader(
            _TextDataset(sentences, self.tokenizer, self.max_len),
            batch_size=self.batch_size,
        )
        preds: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        return preds

    def flag(self, sentences: list[str]) -> list[bool]:
        """Return True for each check-worthy sentence."""
        return [p == 1 for p in self.predict(sentences)]

    @property
    def slug(self) -> str:
        return os.path.basename(self.model_path.rstrip("/\\"))

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from os.path import join
    from sklearn.metrics import classification_report

    multicw = pd.read_csv(join("data", "MultiCW", "multicw-test.csv"))

    multicw["label"] = (pd.to_numeric(multicw["label"], errors="coerce").fillna(0).astype(np.int32))
    multicw["text"] = (multicw["text"].fillna("").astype(str))
    multicw = multicw[multicw["text"].str.strip() != ""].reset_index(drop=True)
    texts = multicw["text"].tolist()
    preds = CheckWorthinessDetector(model_path=join("models", "mdb-multicw")).predict(texts)
    report = classification_report(multicw["label"].to_numpy(), np.array(preds), digits=3,)

    print(report)
