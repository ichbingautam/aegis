"""Neural network architectures for Aegis.

This module provides actor-critic network implementations using
Flax (Linen API) for JAX-based training.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


# Type aliases
Array = jax.Array
PRNGKey = jax.Array
Dtype = Any


class CNNEncoder(nn.Module):
    """Convolutional encoder for image observations.

    Standard architecture following Nature DQN / IMPALA style:
    - 3 convolutional layers with ReLU activation
    - Flattened output for downstream processing

    Attributes:
        channels: Number of output channels for each conv layer
        kernels: Kernel sizes for each conv layer
        strides: Stride sizes for each conv layer
        activation: Activation function
        dtype: Data type for computations
    """

    channels: Sequence[int] = (32, 64, 64)
    kernels: Sequence[int] = (8, 4, 3)
    strides: Sequence[int] = (4, 2, 1)
    activation: Callable = nn.relu
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Encode image observations.

        Args:
            x: Image tensor of shape [..., H, W, C]

        Returns:
            Flattened feature vector
        """
        # Normalize pixel values to [0, 1]
        x = x.astype(self.dtype) / 255.0

        # Apply conv layers
        for channels, kernel, stride in zip(self.channels, self.kernels, self.strides):
            x = nn.Conv(
                features=channels,
                kernel_size=(kernel, kernel),
                strides=(stride, stride),
                padding='VALID',
                dtype=self.dtype,
            )(x)
            x = self.activation(x)

        # Flatten spatial dimensions
        x = x.reshape((*x.shape[:-3], -1))

        return x


class MLPEncoder(nn.Module):
    """MLP encoder for vector observations.

    Attributes:
        hidden_sizes: Sizes of hidden layers
        activation: Activation function
        dtype: Data type for computations
    """

    hidden_sizes: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Encode vector observations.

        Args:
            x: Vector tensor of shape [..., D]

        Returns:
            Encoded feature vector
        """
        x = x.astype(self.dtype)

        for size in self.hidden_sizes:
            x = nn.Dense(size, dtype=self.dtype)(x)
            x = self.activation(x)

        return x


class PolicyHead(nn.Module):
    """Policy head for discrete or continuous actions.

    For discrete actions: outputs logits for categorical distribution
    For continuous actions: outputs mean and log_std for Gaussian

    Attributes:
        action_dim: Dimension of action space
        continuous: Whether actions are continuous
        hidden_sizes: Sizes of hidden layers before output
        activation: Activation function
        log_std_min: Minimum log standard deviation (continuous)
        log_std_max: Maximum log standard deviation (continuous)
    """

    action_dim: int
    continuous: bool = False
    hidden_sizes: Sequence[int] = ()
    activation: Callable = nn.relu
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x: Array) -> distrax.Distribution:
        """Compute action distribution.

        Args:
            x: Feature vector from encoder

        Returns:
            Action distribution (Categorical or MultivariateNormalDiag)
        """
        # Hidden layers
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = self.activation(x)

        if self.continuous:
            # Continuous: output mean and log_std
            mean = nn.Dense(self.action_dim, name='mean')(x)

            # Learnable log_std (state-independent)
            log_std = self.param(
                'log_std',
                nn.initializers.zeros,
                (self.action_dim,)
            )
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            std = jnp.exp(log_std)

            return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        else:
            # Discrete: output logits
            logits = nn.Dense(self.action_dim, name='logits')(x)
            return distrax.Categorical(logits=logits)


class ValueHead(nn.Module):
    """Value head for state value estimation.

    Attributes:
        hidden_sizes: Sizes of hidden layers before output
        activation: Activation function
    """

    hidden_sizes: Sequence[int] = ()
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Compute state value.

        Args:
            x: Feature vector from encoder

        Returns:
            Scalar value estimate
        """
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = self.activation(x)

        value = nn.Dense(1, name='value')(x)
        return jnp.squeeze(value, axis=-1)


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network.

    Architecture:
    - Encoder (CNN or MLP based on observation shape)
    - Shared feature representation
    - Separate policy head (actor)
    - Separate value head (critic)

    Attributes:
        action_dim: Dimension of action space
        continuous: Whether actions are continuous
        encoder: Type of encoder ('cnn' or 'mlp')
        hidden_sizes: Sizes of shared hidden layers
        policy_hidden: Sizes of policy head hidden layers
        value_hidden: Sizes of value head hidden layers
        cnn_channels: CNN encoder channels
        cnn_kernels: CNN encoder kernel sizes
        cnn_strides: CNN encoder stride sizes
        activation: Activation function name
    """

    action_dim: int
    continuous: bool = False
    encoder: str = 'mlp'
    hidden_sizes: Sequence[int] = (256, 256)
    policy_hidden: Sequence[int] = ()
    value_hidden: Sequence[int] = ()
    cnn_channels: Sequence[int] = (32, 64, 64)
    cnn_kernels: Sequence[int] = (8, 4, 3)
    cnn_strides: Sequence[int] = (4, 2, 1)
    activation: str = 'relu'

    def setup(self):
        """Initialize sub-modules."""
        # Get activation function
        act_fn = {
            'relu': nn.relu,
            'tanh': jnp.tanh,
            'elu': nn.elu,
            'swish': nn.swish,
        }.get(self.activation, nn.relu)

        # Encoder
        if self.encoder == 'cnn':
            self.encoder_net = CNNEncoder(
                channels=self.cnn_channels,
                kernels=self.cnn_kernels,
                strides=self.cnn_strides,
                activation=act_fn,
            )
        else:
            self.encoder_net = MLPEncoder(
                hidden_sizes=self.hidden_sizes,
                activation=act_fn,
            )

        # Shared layers (only for CNN, MLP encoder already has them)
        if self.encoder == 'cnn':
            self.trunk = MLPEncoder(
                hidden_sizes=self.hidden_sizes,
                activation=act_fn,
            )
        else:
            self.trunk = None

        # Policy head
        self.policy = PolicyHead(
            action_dim=self.action_dim,
            continuous=self.continuous,
            hidden_sizes=self.policy_hidden,
            activation=act_fn,
        )

        # Value head
        self.value = ValueHead(
            hidden_sizes=self.value_hidden,
            activation=act_fn,
        )

    def __call__(self, obs: Array) -> Tuple[distrax.Distribution, Array]:
        """Forward pass.

        Args:
            obs: Observations of shape [..., *obs_shape]

        Returns:
            Tuple of (action_distribution, value_estimate)
        """
        # Encode observations
        features = self.encoder_net(obs)

        # Shared trunk (for CNN)
        if self.trunk is not None:
            features = self.trunk(features)

        # Policy and value heads
        policy_dist = self.policy(features)
        value = self.value(features)

        return policy_dist, value

    def get_action_and_value(
        self,
        obs: Array,
        key: Optional[PRNGKey] = None,
        deterministic: bool = False,
    ) -> Tuple[Array, Array, Array, Array]:
        """Get action, log_prob, entropy, and value.

        Convenience method for rollout collection.

        Args:
            obs: Observations
            key: PRNG key for sampling (required if not deterministic)
            deterministic: If True, use mode instead of sampling

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        policy_dist, value = self(obs)

        if deterministic:
            action = policy_dist.mode()
        else:
            action = policy_dist.sample(seed=key)

        log_prob = policy_dist.log_prob(action)
        entropy = policy_dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs: Array,
        actions: Array,
    ) -> Tuple[Array, Array, Array]:
        """Evaluate actions under current policy.

        Used during learning to compute policy ratios.

        Args:
            obs: Observations
            actions: Actions to evaluate

        Returns:
            Tuple of (log_prob, entropy, value)
        """
        policy_dist, value = self(obs)

        log_prob = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy()

        return log_prob, entropy, value


def create_network(
    observation_shape: Tuple[int, ...],
    action_dim: int,
    continuous: bool = False,
    config: Optional[dict] = None,
) -> ActorCriticNetwork:
    """Factory function to create actor-critic network.

    Args:
        observation_shape: Shape of observations
        action_dim: Dimension of action space
        continuous: Whether actions are continuous
        config: Optional configuration dictionary

    Returns:
        ActorCriticNetwork instance
    """
    config = config or {}

    # Determine encoder type from observation shape
    if len(observation_shape) >= 3:
        encoder = 'cnn'
    else:
        encoder = 'mlp'

    return ActorCriticNetwork(
        action_dim=action_dim,
        continuous=continuous,
        encoder=config.get('encoder', encoder),
        hidden_sizes=tuple(config.get('hidden_sizes', [256, 256])),
        policy_hidden=tuple(config.get('policy_head_layers', [])),
        value_hidden=tuple(config.get('value_head_layers', [])),
        cnn_channels=tuple(config.get('cnn_channels', [32, 64, 64])),
        cnn_kernels=tuple(config.get('cnn_kernels', [8, 4, 3])),
        cnn_strides=tuple(config.get('cnn_strides', [4, 2, 1])),
        activation=config.get('activation', 'relu'),
    )


def init_network(
    network: ActorCriticNetwork,
    observation_shape: Tuple[int, ...],
    key: PRNGKey,
) -> dict:
    """Initialize network parameters.

    Args:
        network: Network module
        observation_shape: Shape of observations
        key: PRNG key for initialization

    Returns:
        Initialized parameter dictionary
    """
    dummy_obs = jnp.zeros((1,) + observation_shape)
    params = network.init(key, dummy_obs)
    return params
