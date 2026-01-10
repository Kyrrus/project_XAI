from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaConfig:
    """Default config for Ollama LLM calls."""
    host: str = "http://localhost:11434"
    model: str = "gpt-oss:20b"
    timeout_s: float = 60.0


# Custom exception for Ollama-related errors.
# For now not implemented to see in the future if we want to catch it separately
class OllamaError(RuntimeError):
    pass


def ollama_generate(*, cfg: OllamaConfig, prompt: str) -> str:
    """Call Ollama's /api/generate (non-streaming) and return response text."""

    url = cfg.host.rstrip("/") + "/api/generate"
    try:
        resp = requests.post(
            url,
            json={
                "model": cfg.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=cfg.timeout_s,
        )
    except requests.RequestException as e:
        raise OllamaError(
            f"Could not reach Ollama at {cfg.host}. Is it running (ollama serve)? Details: {e}"
        ) from e

    if resp.status_code != 200:
        raise OllamaError(f"Ollama error {resp.status_code}: {resp.text[:500]}")

    data: dict[str, Any] = resp.json() if resp.content else {}
    text = str(data.get("response", "")).strip()
    if not text:
        raise OllamaError("Ollama returned an empty response.")
    return text


def build_llm_explanation_prompt(
    *,
    task_name: str,
    model_key: str,
    class_names: list[str],
    probs: dict[str, float],
    predicted_class: str,
    confidence_margin: float,
    input_kind: str,
    xai_methods: list[str],
    xai_summaries: dict[str, Any] | None = None,
) -> str:
    """Build a prompt from the app's already-computed outputs."""

    # Keep the prompt grounded: mentionned no image access, no medical claims.
    lines: list[str] = [
            "You are an assistant embedded in a local demo app for explainable AI.",
            "Generate a short explanation of the model output using ONLY the structured information provided.",
            "Do not claim you can see the input image/audio. Do not invent visual evidence.",
            "Do not provide medical diagnosis. Use cautious language.",
            "",
            f"Task: {task_name}",
            f"Input type: {input_kind}",
            f"Model key: {model_key}",
            f"Classes: {class_names}",
            f"Predicted class: {predicted_class}",
            f"Probabilities: {probs}",
            f"Confidence margin (|p1-p0|): {confidence_margin:.4f}",
            f"XAI methods available in UI: {xai_methods}",
        ]

    # if the xai has been generated
    if xai_summaries:
        lines.extend(
            [
                "",
                "XAI numeric summaries (derived from the explanation maps; NOT from direct image inspection):",
                f"{xai_summaries}",
            ]
        )

    # final instruction on the format
    lines.extend(
        [
            "",
            "Write 5-8 bullet points:",
            "- 2-3 bullets interpreting the probabilities/margin (confidence, ambiguity).",
            "- 2-3 bullets explaining how to read the XAI methods in this UI.",
            "- If XAI summaries are provided, reference them cautiously (as heuristics).",
            "- 1-2 bullets listing limitations and responsible-use notes.",
        ]
    )

    return "\n".join(lines)
