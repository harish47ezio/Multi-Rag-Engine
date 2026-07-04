import logging

from common.bootstrap import bootstrap
from common.log_utils import preview
from llm import LLMFactory

bootstrap("INFO")
logger = logging.getLogger(__name__)

llm = LLMFactory.create({
    "provider": "ollama",
    "model": "qwen3-coder:480b",
    "base_url": "https://ollama.com",
})

logger.info("alpha_test llm available=%s", llm.is_available())
resp = llm.complete("What model are you in one word!")
logger.info(
    "alpha_test llm response model=%s tokens=%s text='%s'",
    resp.model,
    resp.tokens_used,
    preview(resp.text),
)
