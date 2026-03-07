"""
capabilities/classification/vlm_classifier.py
==============================================
Vision Language Model (VLM) Classifier

Uses a cloud VLM (Gemini Flash by default) to classify and reason about
plant images using natural language. Unlike CLIP, the VLM can:

  - Explain WHY it made a classification decision
  - Answer multiple questions in a single API call
  - Handle complex, multi-factor judgements
  - Return structured JSON with arbitrary fields

Cost: ~$0.001–$0.004 per call depending on model.

Supported backends
------------------
  "gemini-2.0-flash"   ~$0.001/call  Fast, excellent accuracy  (default)
  "gemini-2.0-pro"     ~$0.004/call  Highest accuracy, complex reasoning
  "gpt-4o"             ~$0.003/call  Alternative, strong spatial reasoning

Setup
-----
    export GEMINI_API_KEY=your-key-here
    # or
    export OPENAI_API_KEY=your-key-here

Usage
-----
    from capabilities.classification.vlm_classifier import VLMClassifier

    vlm = VLMClassifier()

    # Simple classification with reasoning
    result = vlm.classify(
        "plant.jpg",
        question="What is the health status of this plant?",
        labels=["healthy", "mildly stressed", "severely stressed", "diseased"],
    )
    print(result.top_label)   # "mildly stressed"
    print(result.reasoning)   # "The lower leaves show slight yellowing..."
    print(result.confidence)  # 0.82

    # Free-form analysis (no fixed labels)
    result = vlm.analyse(
        "plant.jpg",
        prompt="Describe the plant's growth stage, health, and any visible issues.",
    )
    print(result.analysis_text)  # Free text response

    # Structured multi-field analysis
    result = vlm.analyse_structured(
        "plant.jpg",
        schema={
            "growth_stage": "seedling | vegetative | flowering | fruiting",
            "health_score": "0.0 to 1.0",
            "issues": "list of visible problems",
            "recommendation": "one actionable sentence",
        }
    )
    print(result.structured_data)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from capabilities.base import BaseCapability, CapabilityResult


# Mapping of file extensions to MIME types
_MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".webp": "image/webp",
    ".bmp": "image/bmp",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VLMClassificationResult(CapabilityResult):
    """
    Result from VLMClassifier.classify() — labelled classification with reasoning.

    Attributes
    ----------
    question : str
        The question that was asked.
    labels : list of str
        The candidate labels the model chose from.
    top_label : str
        The label the model selected.
    confidence : float
        Model's self-reported confidence (0.0–1.0).
    reasoning : str
        Natural language explanation of the decision.
    all_scores : dict
        Per-label scores if the model provided them (optional).
    """
    question: str = ""
    labels: List[str] = field(default_factory=list)
    top_label: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    all_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "question": self.question,
            "labels": self.labels,
            "top_label": self.top_label,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<VLMClassificationResult label='{self.top_label}' "
            f"conf={self.confidence:.2f} model={self.model_name}>"
        )


@dataclass
class VLMAnalysisResult(CapabilityResult):
    """
    Result from VLMClassifier.analyse() — free-form or structured analysis.

    Attributes
    ----------
    prompt : str
        The prompt that was sent to the model.
    analysis_text : str
        Free-form text response from the model.
    structured_data : dict
        Parsed structured fields (populated by analyse_structured()).
    """
    prompt: str = ""
    analysis_text: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "prompt": self.prompt,
            "analysis_text": self.analysis_text,
            "structured_data": self.structured_data,
        })
        return base

    def __repr__(self) -> str:
        preview = self.analysis_text[:60] + "..." if len(self.analysis_text) > 60 else self.analysis_text
        return f"<VLMAnalysisResult model={self.model_name} text='{preview}'>"


# ---------------------------------------------------------------------------
# VLM Classifier
# ---------------------------------------------------------------------------

class VLMClassifier(BaseCapability):
    """
    Vision Language Model classifier and analyser.

    Sends images to a cloud VLM API with structured prompts and parses
    the response into typed result objects.

    Parameters
    ----------
    model : str
        VLM model to use. Default: 'gemini-2.0-flash'.
    api_key : str or None
        API key. If None, reads from GEMINI_API_KEY environment variable.
    backend : str
        API backend. Currently supports: 'gemini' (default).
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        backend: str = "gemini",
    ) -> None:
        self._model_id = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._backend = backend

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return self._model_id.replace("-", "_").replace(".", "_")

    def run(self, image_path: str, **kwargs) -> VLMAnalysisResult:
        """Alias for analyse(). Pass prompt= as a keyword argument."""
        prompt = kwargs.get("prompt", "Describe this plant image.")
        return self.analyse(image_path, prompt=prompt)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        image_path: str,
        question: str,
        labels: List[str],
    ) -> VLMClassificationResult:
        """
        Ask the VLM to classify the image into one of the provided labels.

        The model is prompted to return structured JSON so the response
        can be parsed reliably.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        question : str
            The classification question, e.g. "What is the health status?"
        labels : list of str
            Candidate labels the model must choose from.

        Returns
        -------
        VLMClassificationResult
        """
        self.validate_image(image_path)
        self._validate_key()

        labels_str = "\n".join(f"  - {l}" for l in labels)
        prompt = f"""{question}

Choose exactly ONE of these labels:
{labels_str}

Respond ONLY with valid JSON — no markdown, no preamble:
{{
  "label": "<chosen label exactly as written above>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence explaining your choice based on visual evidence>"
}}"""

        with self._timer() as t:
            raw = self._call_api(image_path, prompt)

        parsed = self._parse_json(raw, image_path)

        # Validate returned label is in our list
        returned_label = parsed.get("label", "")
        if returned_label not in labels:
            # Fuzzy match: find closest label
            returned_label = min(labels, key=lambda l: _edit_distance(l, returned_label))

        return VLMClassificationResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            question=question,
            labels=labels,
            top_label=returned_label,
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=parsed.get("reasoning", ""),
        )

    def analyse(
        self,
        image_path: str,
        prompt: str,
    ) -> VLMAnalysisResult:
        """
        Send a free-form prompt with the image and return the text response.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str
            Free-form instruction or question about the image.

        Returns
        -------
        VLMAnalysisResult
        """
        self.validate_image(image_path)
        self._validate_key()

        with self._timer() as t:
            raw = self._call_api(image_path, prompt)

        return VLMAnalysisResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            prompt=prompt,
            analysis_text=raw.strip(),
        )

    def analyse_structured(
        self,
        image_path: str,
        schema: Dict[str, str],
    ) -> VLMAnalysisResult:
        """
        Ask the VLM to fill in a structured schema about the image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        schema : dict
            Maps field name → description/type hint.
            Example:
                {
                    "growth_stage":  "one of: seedling, vegetative, flowering, fruiting",
                    "health_score":  "float 0.0 (dead) to 1.0 (perfect health)",
                    "visible_issues": "list of strings, empty list if none",
                    "recommendation": "one actionable sentence for the grower",
                }

        Returns
        -------
        VLMAnalysisResult
            result.structured_data contains the parsed dict.
        """
        self.validate_image(image_path)
        self._validate_key()

        # Build the schema description for the prompt
        schema_lines = "\n".join(f'  "{k}": {v}' for k, v in schema.items())
        empty_json = json.dumps({k: "..." for k in schema.keys()}, indent=2)

        prompt = f"""Analyse this plant image and fill in the following fields.

Respond ONLY with valid JSON matching this schema — no markdown, no extra text:
{empty_json}

Field descriptions:
{schema_lines}"""

        with self._timer() as t:
            raw = self._call_api(image_path, prompt)

        parsed = self._parse_json(raw, image_path)

        return VLMAnalysisResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            prompt=prompt,
            analysis_text=raw,
            structured_data=parsed,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _call_api(self, image_path: str, prompt: str) -> str:
        """Send image + prompt to the configured backend. Returns raw text."""
        if self._backend == "gemini":
            return self._call_gemini(image_path, prompt)
        raise NotImplementedError(f"Backend '{self._backend}' not implemented.")

    def _call_gemini(self, image_path: str, prompt: str) -> str:
        """Call Gemini API and return raw response text."""
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model_id)
        ext = os.path.splitext(image_path)[1].lower()
        mime = _MIME_MAP.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = model.generate_content([
            {"mime_type": mime, "data": image_bytes},
            prompt,
        ])
        return response.text

    def _validate_key(self) -> None:
        """Raise EnvironmentError if no API key is available."""
        if not self._api_key:
            raise EnvironmentError(
                f"[VLMClassifier] No API key configured for backend '{self._backend}'.\n"
                "  Set GEMINI_API_KEY environment variable or pass api_key= to constructor."
            )

    @staticmethod
    def _parse_json(raw: str, image_path: str) -> dict:
        """Strip markdown fences and parse JSON response."""
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[VLMClassifier] Could not parse response for '{image_path}'.\n"
                f"  Raw: {raw[:300]}\n  Error: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance for fuzzy label matching."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]
