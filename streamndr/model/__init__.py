"""Novelty detection models"""
from .minas import Minas
from .ecsminer import ECSMiner
from .ecsminerwf import ECSMinerWF

__all__ = [
    "Minas",
    "ECSMiner",
    "ECSMinerWF",
]