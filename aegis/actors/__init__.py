"""Actors module for Aegis - rollout workers."""

from aegis.actors.rollout_worker import RolloutWorkerLocal

try:
    from aegis.actors.rollout_worker import RolloutWorker
except ImportError:
    RolloutWorker = None

__all__ = [
    "RolloutWorkerLocal",
    "RolloutWorker",
]
