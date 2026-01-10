"""Learners module for Aegis."""

from aegis.learners.vtrace import (
    compute_vtrace,
    compute_vtrace_batch,
    compute_log_rhos,
    should_use_vtrace,
)
from aegis.learners.learner import LearnerLocal

try:
    from aegis.learners.learner import Learner
except ImportError:
    Learner = None

__all__ = [
    "compute_vtrace",
    "compute_vtrace_batch",
    "compute_log_rhos",
    "should_use_vtrace",
    "LearnerLocal",
    "Learner",
]
