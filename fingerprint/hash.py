import hashlib
import logging

logger = logging.getLogger(__name__)


def hash_file(file_path: str) -> str:
    """
    Compute a SHA-256 hash of the file at `file_path`.
    """
    with open(file_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    logger.info("hash_file path=%s sha256=%s", file_path, digest[:12])
    return digest
