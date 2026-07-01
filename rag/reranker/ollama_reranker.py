"""
Reranker-role adapter for Ollama (local or cloud — same HTTP API).

Cross-encoder reranker weights served by Ollama don't share a single
official endpoint, so this adapter supports two scoring strategies:

  * `embed` — call `/api/embed` once per (query, document). Reranker
    models published as "embedding" models on Ollama return a 1-D
    vector whose single value is the relevance score (this is what
    `linux6200/bge-reranker-v2-m3` and similar community ports do).

  * `generate` — call `/api/generate` with a short scoring prompt and
    parse a float out of the response. Useful as a fallback when only
    a generative reranker is available.

The strategy is part of the spec (registry), not auto-detected — the
operator who registered the model knows which API it speaks.

Purely transport: delegates HTTP to the shared `OllamaClient`. Model
identity and strategy are passed in at construction time by the
factory.
"""

import logging
import re
from typing import List

from providers.ollama_client import OllamaClient
from rag.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


class OllamaReranker(BaseReranker):

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        score_strategy: str = "embed",
    ):
        if score_strategy not in ("embed", "generate"):
            raise ValueError(
                f"Unknown OllamaReranker score_strategy '{score_strategy}'. "
                f"Supported: ['embed', 'generate']."
            )
        self.client = client
        self.model = model
        self.score_strategy = score_strategy
        logger.info(
            "OllamaReranker init model=%s strategy=%s",
            model,
            score_strategy,
        )

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        logger.info(
            "OllamaReranker.score model=%s strategy=%s pairs=%d",
            self.model,
            self.score_strategy,
            len(texts),
        )
        if self.score_strategy == "embed":
            return self._score_via_embed(query, texts)
        return self._score_via_generate(query, texts)

    def _score_via_embed(self, query: str, texts: List[str]) -> List[float]:
        # bge-reranker-style: feed "<query>\n<doc>" as one input; the
        # returned vector's first element is the relevance score.
        inputs = [f"{query}\n{text}" for text in texts]
        vectors = self.client.embed(self.model, inputs)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"OllamaReranker embed returned {len(vectors)} vectors for "
                f"{len(texts)} inputs."
            )
        return [float(v[0]) if v else 0.0 for v in vectors]

    def _score_via_generate(self, query: str, texts: List[str]) -> List[float]:
        scores: List[float] = []
        for text in texts:
            prompt = (
                "Rate the relevance of the following passage to the query "
                "on a scale from 0 (irrelevant) to 1 (perfectly relevant). "
                "Reply with a single floating-point number, nothing else.\n\n"
                f"Query: {query}\n\nPassage: {text}\n\nScore:"
            )
            data = self.client.generate(self.model, prompt)
            response = (data.get("response") or "").strip()
            match = _FLOAT_RE.search(response)
            if match is None:
                logger.warning(
                    "OllamaReranker generate could not parse float; response='%s'",
                    response[:80],
                )
                scores.append(0.0)
            else:
                scores.append(float(match.group(0)))
        return scores

    def model_key(self) -> str:
        return self.model
