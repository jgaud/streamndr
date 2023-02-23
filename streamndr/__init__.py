"""
Stream Novelty Detection for River (StreamNDR) is a Python library for online novelty detection.
StreamNDR aims to enable novelty detection in data streams for Python by implementing various 
algorithms that have been proposed in the literature.
"""
from . import model, utils
from .__version__ import __version__ # noqa: F401

__all__ = [
    "model",
    "utils",
]