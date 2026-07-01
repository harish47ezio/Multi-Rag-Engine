"""
Shared connection-state helpers for provider HTTP clients.

`BaseHTTPClient` centralizes the two things every provider client did
identically: resolving `base_url` / `api_key` from constructor args then
environment variables, and building the Bearer-auth request headers. Each
concrete client passes its own env-var names and default URL, and keeps its
own module logger so log lines stay attributed to the concrete client.
"""

import os
from typing import Dict, Optional, Tuple


class BaseHTTPClient:
    base_url: str
    api_key: Optional[str]

    @staticmethod
    def _resolve_credentials(
        base_url: Optional[str],
        api_key: Optional[str],
        *,
        base_url_env: str,
        api_key_env: str,
        default_base_url: str,
    ) -> Tuple[str, Optional[str]]:
        """Resolve (base_url, api_key) from constructor args, then env, then default.

        The returned base_url has any trailing slash stripped.
        """
        resolved_base_url = base_url or os.environ.get(base_url_env) or default_base_url
        resolved_api_key = api_key or os.environ.get(api_key_env)
        return resolved_base_url.rstrip("/"), resolved_api_key

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
