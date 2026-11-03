#!/usr/bin/env python3
"""Validate saved DINOv2 features from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def main() -> int:
    from deltavision.pretrained_validation import main as validate

    return validate()


if __name__ == "__main__":
    raise SystemExit(main())
