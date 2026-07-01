import logging
import os
import time
import requests
from typing import Optional, List, Dict, Any

from common.log_utils import count_chars, preview

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        resolved_base_url = (
            base_url
            or os.environ.get("HUGGINGFACE_HOST")
            or "https://router.huggingface.co/scaleway/v1/embeddings"
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("HUGGING_FACE_API_KEY")
        logger.info(
            "HuggingFaceClient init base_url=%s auth=%s",
            self.base_url,
            "yes" if self.api_key else "no",
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_available(self) -> bool:
        try:
            response = requests.get(self.base_url, headers=self._headers(), timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.info("is_available failed err=%s", e)
            return False

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.info("embed skipped (empty input)")
            return []

        total_chars = count_chars(texts)
        logger.info(
            "embed model=%s inputs=%d total_chars=%d first='%s'",
            model,
            len(texts),
            total_chars,
            preview(texts[0]),
        )

        payload = {"input": texts,"model": model}
        start = time.perf_counter()
        logger.info("auth headers=%s", self._headers())
        logger.info("payload=%s", payload)
        logger.info("url=%s", self.base_url)

        response = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=240000)
        elapsed = time.perf_counter() - start
        logger.info("response=%s", response.json())   

        if response.status_code != 200:
            logger.info(
                "embed FAIL model=%s status=%d body='%s' elapsed=%.2fs",
                model,
                response.status_code,
                preview(response.text),
                elapsed,
            )
            raise RuntimeError(
                f"HuggingFace embed failed: {response.status_code} - {response.text}"
            )

        data = response.json()
        if isinstance(data, list):
            shape = "raw-list"
            vectors = data
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            shape = "openai"
            items = sorted(data["data"], key=lambda x: x.get("index", 0))
            vectors = [item["embedding"] for item in items]
        elif isinstance(data, dict) and "embeddings" in data:
            shape = "hf-embeddings"
            vectors = data["embeddings"]
        else:
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            logger.info("embed unexpected response keys=%s", keys)
            raise RuntimeError(f"Unexpected HuggingFace response: {data}")

        dim = len(vectors[0]) if vectors else 0
        logger.info(
            "embed done model=%s shape=%s vectors=%d dim=%d elapsed=%.2fs",
            model,
            shape,
            len(vectors),
            dim,
            elapsed,
        )
        return vectors