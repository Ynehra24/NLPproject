#!/usr/bin/env python
"""Backward-compatible wrapper for evader training."""

try:
    from post_generation.app.train_evader_app import main
except ModuleNotFoundError:
    from app.train_evader_app import main


if __name__ == "__main__":
    main()
