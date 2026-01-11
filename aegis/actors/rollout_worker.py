"""Rollout worker (Actor) for distributed data collection.

Actors are CPU workers that:
- Run environment simulations
- Collect trajectories using a local policy copy
- Compute trajectory priorities
- Push data to the replay buffer
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from aegis.core.types import ActorInfo, Trajectory


class RolloutWorkerLocal:
    """Local rollout worker for data collection.

    Collects experience by running the environment and storing
    trajectories. Periodically fetches updated policy from
    the parameter server.

    Attributes:
        actor_id: Unique identifier for this actor
        config: Configuration dictionary
        env: Gymnasium environment
        network: Policy network (JAX/Flax)
        policy_version: Current policy version
    """

    def __init__(
        self,
        actor_id: int,
        config: dict[str, Any],
        env_fn,
        network,
        initial_params: dict[str, Any],
    ):
        """Initialize the rollout worker.

        Args:
            actor_id: Unique identifier
            config: Configuration with rollout settings
            env_fn: Function to create environment
            network: Policy network module
            initial_params: Initial policy parameters
        """
        self.actor_id = actor_id
        self.config = config

        # Create environment(s)
        self.num_envs = config.get("envs_per_actor", 1)
        if self.num_envs > 1:
            from gymnasium.vector import SyncVectorEnv

            self.env = SyncVectorEnv([env_fn for _ in range(self.num_envs)])
            self.vectorized = True
        else:
            self.env = env_fn()
            self.vectorized = False

        # Policy
        self.network = network
        self.params = initial_params
        self.policy_version = 0

        # Rollout settings
        self.trajectory_length = config.get("trajectory_length", 128)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)

        # State
        self.obs, _ = self.env.reset()
        self.episode_returns = np.zeros(self.num_envs if self.vectorized else 1)
        self.episode_lengths = np.zeros(self.num_envs if self.vectorized else 1, dtype=np.int32)

        # Statistics
        self.total_steps = 0
        self.episodes_completed = 0
        self.recent_returns: list[float] = []
        self.recent_lengths: list[int] = []

        # Timing
        self._last_time = time.time()
        self._steps_since_last = 0

    def collect_trajectory(self, rng_key=None) -> Trajectory:
        """Collect a trajectory of experience.

        Args:
            rng_key: Optional JAX PRNG key for action sampling

        Returns:
            Trajectory object with collected experience
        """
        import jax
        import jax.numpy as jnp

        T = self.trajectory_length

        # Storage
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        for _t in range(T):
            # Get action from policy
            obs_jax = jnp.asarray(self.obs)

            if rng_key is not None:
                rng_key, action_key = jax.random.split(rng_key)
            else:
                action_key = jax.random.PRNGKey(int(time.time() * 1e6) % (2**31))

            action, log_prob, _, value = self.network.apply(
                self.params,
                obs_jax,
                action_key,
                method=self.network.get_action_and_value,
            )

            # Convert to numpy
            action_np = np.asarray(action)
            log_prob_np = np.asarray(log_prob)
            value_np = np.asarray(value)

            # Store
            observations.append(self.obs.copy())
            actions.append(action_np)
            log_probs.append(log_prob_np)
            values.append(value_np)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = (
                terminated | truncated
                if isinstance(terminated, np.ndarray)
                else terminated or truncated
            )

            rewards.append(reward)
            dones.append(done)

            # Update episode stats
            if self.vectorized:
                self.episode_returns += reward
                self.episode_lengths += 1
                for i, d in enumerate(done):
                    if d:
                        self.episodes_completed += 1
                        self.recent_returns.append(float(self.episode_returns[i]))
                        self.recent_lengths.append(int(self.episode_lengths[i]))
                        self.episode_returns[i] = 0
                        self.episode_lengths[i] = 0
            else:
                self.episode_returns[0] += reward
                self.episode_lengths[0] += 1
                if done:
                    self.episodes_completed += 1
                    self.recent_returns.append(float(self.episode_returns[0]))
                    self.recent_lengths.append(int(self.episode_lengths[0]))
                    self.episode_returns[0] = 0
                    self.episode_lengths[0] = 0
                    next_obs, _ = self.env.reset()

            self.obs = next_obs
            self.total_steps += 1 if not self.vectorized else self.num_envs

        # Get bootstrap value
        obs_jax = jnp.asarray(self.obs)
        _, bootstrap_value = self.network.apply(self.params, obs_jax)
        bootstrap_value_np = np.asarray(bootstrap_value)

        # Stack arrays
        observations.append(self.obs.copy())  # Bootstrap observation

        if self.vectorized:
            # Flatten vectorized envs
            observations = np.stack(observations, axis=0)  # [T+1, num_envs, ...]
            actions = np.stack(actions, axis=0)
            rewards = np.stack(rewards, axis=0)
            dones = np.stack(dones, axis=0)
            log_probs = np.stack(log_probs, axis=0)
            values = np.stack(values, axis=0)
            values = np.concatenate([values, bootstrap_value_np[None]], axis=0)

            # Reshape to [T * num_envs, ...]
            batch_size = T * self.num_envs
            observations = observations.reshape(T + 1, -1, *observations.shape[3:])
            actions = actions.reshape(batch_size, *actions.shape[2:])
            rewards = rewards.flatten()
            dones = dones.flatten()
            log_probs = log_probs.flatten()
            values = values.reshape(T + 1, -1)
        else:
            observations = np.array(observations)
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones).astype(np.float32)
            log_probs = np.array(log_probs)
            values = np.concatenate([np.array(values), [bootstrap_value_np]])

        # Keep recent stats bounded
        self._trim_recent_stats()

        return Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            values=values,
            policy_version=self.policy_version,
            actor_id=self.actor_id,
        )

    def update_policy(self, params: dict[str, Any], version: int) -> None:
        """Update local policy copy.

        Args:
            params: New policy parameters
            version: Policy version number
        """
        self.params = params
        self.policy_version = version

    def get_info(self) -> ActorInfo:
        """Get actor statistics.

        Returns:
            ActorInfo with current stats
        """
        now = time.time()
        elapsed = now - self._last_time
        steps_per_sec = self._steps_since_last / max(elapsed, 1e-6)
        self._last_time = now
        self._steps_since_last = 0

        mean_return = np.mean(self.recent_returns) if self.recent_returns else 0.0
        mean_length = np.mean(self.recent_lengths) if self.recent_lengths else 0.0

        return ActorInfo(
            actor_id=self.actor_id,
            policy_version=self.policy_version,
            episodes_completed=self.episodes_completed,
            steps_collected=self.total_steps,
            mean_episode_return=float(mean_return),
            mean_episode_length=float(mean_length),
            steps_per_second=steps_per_sec,
        )

    def _trim_recent_stats(self, max_size: int = 100):
        """Trim recent stats lists to bounded size."""
        if len(self.recent_returns) > max_size:
            self.recent_returns = self.recent_returns[-max_size:]
            self.recent_lengths = self.recent_lengths[-max_size:]

    def close(self):
        """Clean up resources."""
        self.env.close()


if RAY_AVAILABLE:

    @ray.remote
    class RolloutWorker:
        """Ray actor for distributed rollout collection.

        Runs on CPU workers and collects experience in parallel.
        Periodically syncs policy with the parameter server.
        """

        def __init__(
            self,
            actor_id: int,
            config: dict[str, Any],
            env_name: str,
            network_config: dict[str, Any],
        ):
            """Initialize the distributed rollout worker.

            Args:
                actor_id: Unique identifier
                config: Configuration dictionary
                env_name: Gymnasium environment name
                network_config: Network architecture config
            """
            import gymnasium as gym
            import jax

            from aegis.networks import create_network, init_network

            self.actor_id = actor_id
            self.config = config

            # Create environment factory
            def env_fn():
                return gym.make(env_name)

            # Get env specs
            test_env = env_fn()
            obs_shape = test_env.observation_space.shape
            if hasattr(test_env.action_space, "n"):
                action_dim = test_env.action_space.n
                continuous = False
            else:
                action_dim = test_env.action_space.shape[0]
                continuous = True
            test_env.close()

            # Create network
            self.network = create_network(
                observation_shape=obs_shape,
                action_dim=action_dim,
                continuous=continuous,
                config=network_config,
            )

            # Initialize parameters
            rng = jax.random.PRNGKey(actor_id)
            initial_params = init_network(self.network, obs_shape, rng)

            # Create local worker
            self._worker = RolloutWorkerLocal(
                actor_id=actor_id,
                config=config,
                env_fn=env_fn,
                network=self.network,
                initial_params=initial_params,
            )

            # PRNG key
            self._rng = jax.random.PRNGKey(actor_id + 1000)

        def collect_trajectory(self) -> Trajectory:
            """Collect a trajectory."""
            import jax

            self._rng, key = jax.random.split(self._rng)
            return self._worker.collect_trajectory(key)

        def update_policy(self, params: dict[str, Any], version: int) -> None:
            """Update local policy."""
            self._worker.update_policy(params, version)

        def get_info(self) -> ActorInfo:
            """Get actor info."""
            return self._worker.get_info()

        def close(self):
            """Clean up."""
            self._worker.close()
