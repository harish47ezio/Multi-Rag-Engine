import logging

from dotenv import load_dotenv

load_dotenv()

import truststore

truststore.inject_into_ssl()

from common.log_utils import preview, setup_logging
from llm import LLMFactory

setup_logging("INFO")
logger = logging.getLogger(__name__)

llm = LLMFactory.create({
    "provider": "ollama",
    "model": "qwen3-next:80b",
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
