import logging
import time
import requests
from typing import Optional, List

from common.log_utils import count_chars, preview
from providers.base_client import BaseHTTPClient

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseHTTPClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url, self.api_key = self._resolve_credentials(
            base_url,
            api_key,
            base_url_env="HUGGINGFACE_HOST",
            api_key_env="HUGGING_FACE_API_KEY",
            default_base_url="https://router.huggingface.co/scaleway/v1/embeddings",
        )
        logger.info(
            "HuggingFaceClient init base_url=%s auth=%s",
            self.base_url,
            "yes" if self.api_key else "no",
        )

    def is_available(self) -> bool:
        """Reachability probe.

        The endpoint is a POST-only, OpenAI-style embeddings route, so a GET
        will not return 200 even when the host is perfectly healthy. We only
        treat genuine transport failures (no connection, timeout) as
        unreachable; any HTTP response — including 401/404/405 — means the
        host is up and is considered available.
        """
        try:
            requests.get(self.base_url, headers=self._headers(), timeout=10)
            return True
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.info("is_available unreachable err=%s", e)
            return False
        except Exception as e:
            logger.info("is_available unexpected err=%s", e)
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

        payload = {"input": texts, "model": model}
        start = time.perf_counter()

        response = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=240)
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