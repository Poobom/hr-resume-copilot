"""OpenAI client wrapper with cost tracking.

Reused from task01_solution. Extended with batch embedding helper.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

PRICING = {
    "text-embedding-3-small": {"input": 0.020, "output": 0.0},
    "text-embedding-3-large": {"input": 0.130, "output": 0.0},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4.1-nano": {"input": 0.100, "output": 0.400},
    "gpt-4.1-mini": {"input": 0.400, "output": 1.600},
}


@dataclass
class CallRecord:
    step: str
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    timestamp: str


@dataclass
class CostTracker:
    records: list[CallRecord] = field(default_factory=list)

    def log(self, step: str, model: str, tokens_in: int, tokens_out: int) -> float:
        price = PRICING.get(model)
        cost = 0.0 if price is None else (tokens_in * price["input"] + tokens_out * price["output"]) / 1_000_000
        self.records.append(
            CallRecord(
                step=step,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            )
        )
        return cost

    @property
    def total(self) -> float:
        return sum(r.cost for r in self.records)

    def by_step(self) -> dict[str, dict[str, float]]:
        agg: dict[str, dict[str, float]] = {}
        for r in self.records:
            slot = agg.setdefault(r.step, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "model": r.model})
            slot["calls"] += 1
            slot["tokens_in"] += r.tokens_in
            slot["tokens_out"] += r.tokens_out
            slot["cost"] += r.cost
        return agg

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total, 6),
            "by_step": self.by_step(),
            "records": [asdict(r) for r in self.records],
        }

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "cost.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st  # type: ignore
            key = st.secrets.get("OPENAI_API_KEY")  # type: ignore
        except Exception:
            pass
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Set in .env (local) or Streamlit Cloud Secrets (deploy)."
        )
    return OpenAI(api_key=key)


def chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    cost: CostTracker,
    step: str,
    max_tokens: int = 1500,
    temperature: float = 0.2,
    retries: int = 4,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = resp.usage
            cost.log(step, model, usage.prompt_tokens, usage.completion_tokens)
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            last_err = exc
            is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if attempt < retries:
                delay = min(30.0, 2.0 * (2 ** attempt)) if is_rate_limit else 1.0
                time.sleep(delay)
                continue
            raise RuntimeError(f"chat_json failed for {step} after {retries + 1} attempts: {exc}") from last_err
    return {}


def embed_batch(
    client: OpenAI,
    texts: list[str],
    cost: CostTracker,
    step: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
) -> list[list[float]]:
    """Embed a list of texts in batches. Returns embeddings in input order."""
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        usage = resp.usage
        cost.log(step, model, usage.prompt_tokens, 0)
        for item in resp.data:
            all_vecs.append(item.embedding)
    return all_vecs


def vision_ocr(
    client: OpenAI,
    image_paths: list[Path],
    cost: CostTracker,
    step: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 2000,
    retries: int = 4,
) -> str:
    """Extract text from images (scanned PDF pages) via Vision API with rate-limit backoff."""
    import base64

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "다음 이미지에서 이력서 텍스트 전체를 순서대로 추출하세요. 표·레이아웃은 무시하고 글자만 줄바꿈으로 정리. 출력은 원문 텍스트만."}
    ]
    for p in image_paths:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            usage = resp.usage
            cost.log(step, model, usage.prompt_tokens, usage.completion_tokens)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if attempt < retries and is_rate_limit:
                delay = min(30.0, 2.0 * (2 ** attempt))
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"vision_ocr failed after {retries + 1} attempts") from last_err
