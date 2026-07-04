import logging

from common.bootstrap import bootstrap
from rag.factory import MotherFactory

bootstrap("INFO")
logger = logging.getLogger(__name__)

instance = MotherFactory.interactive_pick().embedding
logger.info("alpha_test instance=%s", instance.describe())

instance.embed(["Hello, world"])
logger.info("alpha_test embedder dimension=%d", instance.dimension())
