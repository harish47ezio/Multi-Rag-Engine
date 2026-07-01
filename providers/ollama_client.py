import logging
import time
import requests
from typing import Optional, List, Dict, Any

from common.log_utils import count_chars, preview
from providers.base_client import BaseHTTPClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseHTTPClient):
    """
    HTTP client for Ollama (local) and Ollama Cloud.

    Holds connection-level state only: base_url, api_key, headers.
    Model is NOT bound to the client; it is passed per call by the role adapter.

    Credential resolution (in order):
        base_url: constructor arg -> $OLLAMA_HOST -> "http://localhost:11434"
        api_key:  constructor arg -> $OLLAMA_API_KEY -> None (local Ollama needs no key)

    Note: this client deliberately exposes no tokenize() method. Token counting
    is the responsibility of each role adapter, which uses its model's own
    (fast, local) tokenizer instead of paying a forward pass per call.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url, self.api_key = self._resolve_credentials(
            base_url,
            api_key,
            base_url_env="OLLAMA_HOST",
            api_key_env="OLLAMA_API_KEY",
            default_base_url="http://localhost:11434",
        )
        logger.info(
            "OllamaClient init base_url=%s auth=%s",
            self.base_url,
            "yes" if self.api_key else "no",
        )

    def list_model_names(self, timeout: int = 10) -> set:
        """Return the set of model names served by this endpoint (from /api/tags).

        Raises on any transport/HTTP error so callers can distinguish
        "unreachable" from "reachable but model not served".
        """
        url = f"{self.base_url}/api/tags"
        resp = requests.get(url, headers=self._headers(), timeout=timeout)
        resp.raise_for_status()
        tags = resp.json().get("models", []) or []
        return {m.get("name") or m.get("model") for m in tags}

    def is_available(self) -> bool:
        logger.info("is_available GET %s/api/tags", self.base_url)
        try:
            self.list_model_names(timeout=5)
            logger.info("is_available result=True")
            return True
        except Exception as exc:
            logger.info("is_available unreachable err=%s", exc)
            return False

    def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        logger.info(
            "generate model=%s prompt_chars=%d prompt='%s'",
            model,
            len(prompt),
            preview(prompt),
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        start = time.perf_counter()
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=120)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "generate done model=%s eval_count=%s elapsed=%.2fs",
            model,
            data.get("eval_count"),
            elapsed,
        )
        return data

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.info("embed skipped (empty input)")
            return []

        url = f"{self.base_url}/api/embed"
        total_chars = count_chars(texts)
        logger.info(
            "embed model=%s inputs=%d total_chars=%d first='%s'",
            model,
            len(texts),
            total_chars,
            preview(texts[0]),
        )

        payload = {"model": model, "input": texts}
        start = time.perf_counter()
        response = requests.post(url, json=payload, headers=self._headers(), timeout=240)
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            logger.info(
                "embed FAIL model=%s status=%d body='%s' elapsed=%.2fs",
                model,
                response.status_code,
                preview(response.text),
                elapsed,
            )
            raise RuntimeError(
                f"Ollama embed failed: {response.status_code} - {response.text}"
            )

        data = response.json()
        if "embeddings" not in data:
            logger.info("embed unexpected response keys=%s", list(data.keys()))
            raise RuntimeError(f"Unexpected Ollama response: {data}")

        vectors = data["embeddings"]
        dim = len(vectors[0]) if vectors else 0
        logger.info(
            "embed done model=%s vectors=%d dim=%d elapsed=%.2fs",
            model,
            len(vectors),
            dim,
            elapsed,
        )
        return vectors
