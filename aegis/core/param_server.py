"""Parameter server for distributed policy management.

Manages versioned policy parameters and coordinates between
actors and learners in the distributed training setup.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class ParameterServerLocal:
    """Local (non-distributed) parameter server.

    Used for single-node training and testing.

    Attributes:
        params: Current policy parameters
        version: Current policy version number
        step: Global training step
    """

    def __init__(self, initial_params: Dict[str, Any]):
        """Initialize the parameter server.

        Args:
            initial_params: Initial policy parameters
        """
        self.params = initial_params
        self.version = 0
        self.step = 0
        self._lock = threading.Lock()

    def get_weights(self) -> Tuple[Dict[str, Any], int]:
        """Get current policy weights and version.

        Returns:
            Tuple of (params, version)
        """
        with self._lock:
            return self.params, self.version

    def set_weights(self, params: Dict[str, Any]) -> int:
        """Update policy weights.

        Args:
            params: New policy parameters

        Returns:
            New version number
        """
        with self._lock:
            self.params = params
            self.version += 1
            return self.version

    def get_version(self) -> int:
        """Get current version without weights.

        Returns:
            Current version number
        """
        return self.version

    def increment_step(self, delta: int = 1) -> int:
        """Increment global step counter.

        Args:
            delta: Amount to increment

        Returns:
            New step count
        """
        with self._lock:
            self.step += delta
            return self.step

    def get_step(self) -> int:
        """Get current global step.

        Returns:
            Current step
        """
        return self.step

    def get_info(self) -> Dict[str, Any]:
        """Get parameter server info.

        Returns:
            Dictionary with version, step, and param count
        """
        return {
            'version': self.version,
            'step': self.step,
            'num_params': sum(
                np.prod(v.shape) for v in
                _flatten_params(self.params).values()
            ),
        }


def _flatten_params(params: Dict[str, Any], prefix: str = '') -> Dict[str, np.ndarray]:
    """Flatten nested parameter dictionary."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_params(v, key))
        else:
            flat[key] = np.asarray(v)
    return flat


if RAY_AVAILABLE:
    @ray.remote
    class ParameterServer:
        """Ray actor for distributed parameter management.

        Handles concurrent access from multiple actors and learners.
        Uses Ray's object store for efficient parameter distribution.
        """

        def __init__(self, initial_params: Dict[str, Any]):
            """Initialize the distributed parameter server.

            Args:
                initial_params: Initial policy parameters
            """
            self._server = ParameterServerLocal(initial_params)

        def get_weights(self) -> Tuple[Dict[str, Any], int]:
            """Get current policy weights and version."""
            return self._server.get_weights()

        def set_weights(self, params: Dict[str, Any]) -> int:
            """Update policy weights."""
            return self._server.set_weights(params)

        def get_version(self) -> int:
            """Get current version without weights."""
            return self._server.get_version()

        def increment_step(self, delta: int = 1) -> int:
            """Increment global step counter."""
            return self._server.increment_step(delta)

        def get_step(self) -> int:
            """Get current global step."""
            return self._server.get_step()

        def get_info(self) -> Dict[str, Any]:
            """Get parameter server info."""
            return self._server.get_info()
