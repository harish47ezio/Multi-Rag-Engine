"""Entrypoint: `python -m rag.registry ...` -> registry CLI."""

import sys

from rag.registry.cli import main

if __name__ == "__main__":
    sys.exit(main())
