from providers.ollama_client import OllamaClient
from .base import BaseLLMAdapter
from .ollama_adapter import OllamaAdapter


class LLMFactory:

    @staticmethod
    def create(config: dict) -> BaseLLMAdapter:
        """
        config:
            {
                "provider": "ollama",
                "model": "llama3.1:8b",                # or a cloud model like "gpt-oss:20b"
                "base_url": "http://localhost:11434",  # optional; falls back to $OLLAMA_HOST, then localhost
                "api_key": "ollama_xxx",               # optional; falls back to $OLLAMA_API_KEY
            }
        """
        provider = config.get("provider", "ollama").lower()

        if provider == "ollama":
            client = OllamaClient(
                base_url=config.get("base_url"),
                api_key=config.get("api_key"),
            )
            return OllamaAdapter(client=client, model=config["model"])

        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: ['ollama']"
        )
