import logging

from common.bootstrap import bootstrap
from rag.factory import EmbedderFactory

bootstrap("INFO")
logger = logging.getLogger(__name__)

instance = EmbedderFactory.interactive_pick()
logger.info("alpha_test instance=%s", instance.describe())

instance.embed(["Hello, world"])
logger.info("alpha_test embedder dimension=%d", instance.dimension())
