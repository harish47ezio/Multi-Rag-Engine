import logging

import truststore

truststore.inject_into_ssl()

from common.log_utils import setup_logging
from rag.factory import EmbedderFactory

setup_logging("INFO")
logger = logging.getLogger(__name__)

instance = EmbedderFactory.interactive_pick()
logger.info("alpha_test instance=%s", instance.describe())

instance.embed(["Hello, world"])
logger.info("alpha_test embedder dimension=%d", instance.dimension())
