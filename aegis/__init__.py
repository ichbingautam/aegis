"""Aegis: Scalable Distributed Reinforcement Learning Framework.

A high-throughput distributed RL system for experimentally evaluating
scaling laws in on-policy algorithms.

Key Features:
- Actor-learner separation (IMPALA-inspired)
- Distributed Prioritized Experience Replay
- V-trace off-policy correction
- Algorithm-agnostic (PPO, V-MPO)
- >1M env steps/sec scaling

Example:
    >>> from aegis import train
    >>> train.main()
"""

__version__ = "0.1.0"
__author__ = "Gautam Shubham"

from aegis.algorithms import create_algorithm
from aegis.core.types import Batch, Trajectory

__all__ = [
    "__version__",
    "Batch",
    "Trajectory",
    "create_algorithm",
]
