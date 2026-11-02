"""Frozen B1 scoring. Importing this package does not load training code or data."""

from .b1_inference import B1Model, load_model

__all__ = ["B1Model", "load_model"]
