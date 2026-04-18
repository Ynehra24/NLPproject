#!/usr/bin/env python
"""Backward-compatible wrapper for detector training."""

try:
    from post_generation.app.train_detector_app import main
except ModuleNotFoundError:
    from app.train_detector_app import main


if __name__ == "__main__":
    main()
