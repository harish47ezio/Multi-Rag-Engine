import logging

from common.log_utils import preview
from providers.ollamaClient import OllamaClient

from .base import BaseLLMAdapter, LLMResponse

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseLLMAdapter):
    """Completion-role adapter for Ollama. Delegates HTTP to OllamaClient."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model
        logger.info("LLM OllamaAdapter init model=%s", model)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        logger.info(
            "LLM complete model=%s prompt_chars=%d prompt='%s'",
            self.model,
            len(prompt),
            preview(prompt),
        )
        data = self.client.generate(self.model, prompt, **kwargs)
        response = LLMResponse(
            text=data["response"],
            model=self.model,
            provider="ollama",
            tokens_used=data.get("eval_count"),
        )
        logger.info(
            "LLM complete done model=%s tokens=%s response='%s'",
            self.model,
            response.tokens_used,
            preview(response.text),
        )
        return response

    def is_available(self) -> bool:
        return self.client.is_available()
