"""
One-call setup for entry-point scripts.

`bootstrap()` bundles the three things every runnable script did by hand:
load `.env`, inject the system trust store into TLS (needed on corporate
networks), and configure logging. Library modules must NOT call this — it is
for `__main__`-style entry points only.
"""

from dotenv import load_dotenv
import truststore

from common.log_utils import setup_logging


def bootstrap(level: str = "INFO") -> None:
    """Load env vars, inject the system trust store, and configure logging."""
    load_dotenv()
    truststore.inject_into_ssl()
    setup_logging(level)
