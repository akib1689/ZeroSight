#!/usr/bin/env python3
"""Derive content-class disclosure metrics (thin shim).

Implementation lives in ``scripts/disclosure_metrics/``; see that package's
``__init__`` for the content-class model and ``report.py`` for the split
output layout (``derived/{common,dr,vvf,rd,maxleak}.json``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_metrics.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
