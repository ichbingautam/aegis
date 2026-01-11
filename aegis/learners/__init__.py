"""Learners module for Aegis."""

from aegis.learners.learner import LearnerLocal
from aegis.learners.vtrace import (
    compute_log_rhos,
    compute_vtrace,
    compute_vtrace_batch,
    should_use_vtrace,
)

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
