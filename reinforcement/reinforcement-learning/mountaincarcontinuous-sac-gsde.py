"""
TensorFlow SAC + gSDE (corrected & stabilized) - Ported from PyTorch

This file is a TensorFlow port of a PyTorch implementation of SAC with gSDE.
It aims to be a correct, stable version that:
  - Implements gSDE actor with pre-tanh sampling and analytic log-prob.
  - Applies tanh squashing with change-of-variable correction to log-probs.
  - Uses proper SAC update: target uses entropy term, actor loss minimizes
    (alpha * logp - Q). Supports automatic entropy tuning (recommended).
  - Logs step count, episode returns, sigma and alpha for debugging.
  - Adds observation normalization helper (important for MountainCarContinuous).

Key hyperparameters you can tune for MountainCarContinuous-v0:
  - cfg.use_gsde: True/False
  - cfg.sde_sample_freq: set -1 to keep E fixed during episode (often helps)
  - cfg.log_std_init: initial log-std (try -2.0..-0.5)
  - cfg.batch_size, lr, alpha target
  - obs normalization is enabled by default

This implementation is intentionally explicit and educational; it's not as
optimized as production libraries, but it fixes several issues that caused
instability earlier (missing log-prob correction, missing entropy term, etc.).

Run:
  python your_tensorflow_sac_filename.py

"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# ==========================
# Config
# ==========================
@dataclass
class SACConfig:
    env_id: str = "MountainCarContinuous-v0"
    seed: int = 0
    total_steps: int = 200_000
    start_steps: int = 1000
    replay_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    # automatic entropy tuning
    target_entropy: Optional[float] = 2.0 # None  # if None uses -action_dim
    use_auto_alpha: bool = True
    init_alpha: float = 0.2
    # gSDE
    use_gsde: bool = True
    sde_sample_freq: int = -1  # -1: E fixed during episode; helps stability
    feature_dim: int = 64
    log_std_init: float = 0.5 # -1.0
    # obs normalization
    obs_normalize: bool = True

# ==========================
# Utils
# ==========================

def mlp(sizes, activation='relu', out_act=None):
    layers_list = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers_list.append(layers.Dense(sizes[i+1], activation=act))
    return keras.Sequential(layers_list)

# Simple running mean/std for observation normalization (remains in NumPy)
class RunningStat:
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape, np.float64)
        self._S = np.zeros(shape, np.float64)

    def push(self, x):
        x = np.asarray(x)
        if x.shape != self._M.shape:
            raise ValueError("shape mismatch")
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return (self._S / (self._n - 1)) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

# ==========================
# gSDE noise module
# ==========================
class gSDENoise(layers.Layer):
    def __init__(self, action_dim: int, feature_dim: int, log_std_init: float = -1.0, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        print(log_std_init)
        self.log_std = tf.Variable(
            initial_value=tf.ones(action_dim) * log_std_init,
            trainable=True,
            name="log_std"
        )
        self.E = tf.Variable(
            initial_value=tf.random.normal((self.feature_dim, self.action_dim)),
            trainable=False,
            name="E"
        )
        self.step_count = tf.Variable(0, trainable=False, dtype=tf.int64)

    @property
    def sigma(self) -> tf.Tensor:
        return tf.exp(self.log_std)

    def maybe_resample(self, sde_sample_freq: int):
        if sde_sample_freq == -1:
            return
        if (self.step_count % max(1, sde_sample_freq)) == 0:
            self.E.assign(tf.random.normal(self.E.shape))
        self.step_count.assign_add(1)

    def noise_term(self, phi: tf.Tensor) -> tf.Tensor:
        return tf.matmul(phi, self.E)

# ==========================
# Actor with gSDE
# ==========================
class GSDEActor(Model):
    def __init__(self, obs_dim:int, act_dim:int, cfg:SACConfig, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.feature_net = mlp([obs_dim, 256, cfg.feature_dim], activation='relu')
        self.mu_head = mlp([cfg.feature_dim, 256, act_dim], activation='relu', out_act=None)
        self.use_gsde = cfg.use_gsde
        if self.use_gsde:
            self.noise = gSDENoise(act_dim, cfg.feature_dim, cfg.log_std_init)
        else:
            self.log_std = tf.Variable(
                initial_value=tf.ones(act_dim) * cfg.log_std_init,
                trainable=True,
                name="log_std"
            )

    def call(self, obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.forward_features(obs)

    def forward_features(self, obs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        phi = self.feature_net(obs)
        mu = self.mu_head(phi)
        return mu, phi

    def sample_pre_tanh(self, obs: tf.Tensor, sde_sample_freq: Optional[int] = None, deterministic: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        mu, phi = self.forward_features(obs)
        if self.use_gsde:
            if sde_sample_freq is not None:
                self.noise.maybe_resample(sde_sample_freq)
            if deterministic:
                pre_tanh = mu
            else:
                noise = self.noise.noise_term(phi)
                pre_tanh = mu + noise * self.noise.sigma
            std_state = tf.norm(phi, ord=2, axis=1, keepdims=True)
            std = std_state * tf.reshape(self.noise.sigma, (1, -1))
        else:
            sigma = tf.exp(self.log_std)
            if deterministic:
                pre_tanh = mu
            else:
                pre_tanh = mu + tf.random.normal(tf.shape(mu)) * tf.reshape(sigma, (1, -1))
            std = tf.broadcast_to(tf.reshape(sigma, (1, -1)), tf.shape(mu))

        std = std + 1e-8
        var = tf.square(std)
        log_prob = -0.5 * (tf.square(pre_tanh - mu) / var + 2 * tf.math.log(std) + math.log(2 * math.pi))
        log_prob = tf.reduce_sum(log_prob, axis=1)
        return pre_tanh, log_prob

    def log_prob_from_pre_tanh(self, obs: tf.Tensor, pre_tanh: tf.Tensor) -> tf.Tensor:
        mu, phi = self.forward_features(obs)
        if self.use_gsde:
            std_state = tf.norm(phi, ord=2, axis=1, keepdims=True)
            std = std_state * tf.reshape(self.noise.sigma, (1, -1))
        else:
            std = tf.broadcast_to(tf.reshape(tf.exp(self.log_std), (1, -1)), tf.shape(mu))

        std = std + 1e-8
        var = tf.square(std)
        log_prob = -0.5 * (tf.square(pre_tanh - mu) / var + 2 * tf.math.log(std) + math.log(2 * math.pi))
        return tf.reduce_sum(log_prob, axis=1)

# ==========================
# Squash (tanh) helpers
# ==========================
def squash_and_correct(pre_tanh: tf.Tensor, logp: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    tanh_a = tf.tanh(pre_tanh)
    # log|det Jacobian| = sum log(1 - tanh^2)
    log_det = tf.reduce_sum(tf.math.log(1 - tf.square(tanh_a) + 1e-6), axis=1)
    corrected = logp - log_det
    return tanh_a, corrected

# ==========================
# Q network
# ==========================
class QNetwork(Model):
    def __init__(self, obs_dim:int, act_dim:int, **kwargs):
        super().__init__(**kwargs)
        self.net = mlp([obs_dim + act_dim, 256, 256, 1], activation='relu')

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        obs, act = inputs
        x = tf.concat([obs, act], axis=1)
        q_val = self.net(x)
        return tf.squeeze(q_val, axis=-1)

# ==========================
# Replay buffer (numpy-backed for efficiency)
# ==========================
class ReplayBuffer:
    def __init__(self, obs_dim:int, act_dim:int, size:int):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size:int):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = tf.convert_to_tensor(self.obs_buf[idx], dtype=tf.float32)
        acts = tf.convert_to_tensor(self.act_buf[idx], dtype=tf.float32)
        rews = tf.convert_to_tensor(self.rew_buf[idx], dtype=tf.float32)
        next_obs = tf.convert_to_tensor(self.next_obs_buf[idx], dtype=tf.float32)
        done = tf.convert_to_tensor(self.done_buf[idx], dtype=tf.float32)
        return obs, acts, rews, next_obs, done

# ==========================
# SAC Agent
# ==========================
class SACAgent:
    def __init__(self, env:gym.Env, cfg:SACConfig):
        self.env = env
        self.cfg = cfg
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_low = tf.convert_to_tensor(env.action_space.low, dtype=tf.float32)
        self.action_high = tf.convert_to_tensor(env.action_space.high, dtype=tf.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # networks
        self.actor = GSDEActor(obs_dim, act_dim, cfg)
        self.q1 = QNetwork(obs_dim, act_dim)
        self.q2 = QNetwork(obs_dim, act_dim)
        self.q1_targ = QNetwork(obs_dim, act_dim)
        self.q2_targ = QNetwork(obs_dim, act_dim)

        # Initialize networks with dummy data to build them
        dummy_obs = tf.random.normal((1, obs_dim))
        dummy_act = tf.random.normal((1, act_dim))
        self.actor(dummy_obs)
        self.q1((dummy_obs, dummy_act))
        self.q2((dummy_obs, dummy_act))
        self.q1_targ((dummy_obs, dummy_act))
        self.q2_targ((dummy_obs, dummy_act))

        self.q1_targ.set_weights(self.q1.get_weights())
        self.q2_targ.set_weights(self.q2.get_weights())

        # optimizers
        self.actor_opt = optimizers.Adam(learning_rate=cfg.lr)
        self.q_opt = optimizers.Adam(learning_rate=cfg.lr)

        # entropy alpha (auto)
        if cfg.use_auto_alpha:
            self.log_alpha = tf.Variable(math.log(cfg.init_alpha), trainable=True)
            self.alpha_opt = optimizers.Adam(learning_rate=cfg.lr)
            self.target_entropy = -act_dim if cfg.target_entropy is None else cfg.target_entropy
        else:
            self.log_alpha = None
            self.alpha = tf.constant(cfg.init_alpha, dtype=tf.float32)

        # replay
        self.replay = ReplayBuffer(obs_dim, act_dim, cfg.replay_size)

        # obs normalizer
        if cfg.obs_normalize:
            self.obs_rstat = RunningStat((obs_dim,))
        else:
            self.obs_rstat = None

        self.total_steps = 0

        # Compile the update function for performance
        self.compiled_train_step = tf.function(self._train_step)

    def normalize_obs(self, o: np.ndarray) -> np.ndarray:
        if self.obs_rstat is None:
            return o
        self.obs_rstat.push(o)
        mean = self.obs_rstat.mean
        std = self.obs_rstat.std
        return (o - mean) / (std + 1e-8)

    def scale_action(self, tanh_action: tf.Tensor) -> tf.Tensor:
        return tanh_action * self.action_scale + self.action_bias

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_n = self.normalize_obs(obs.copy()) if self.obs_rstat is not None else obs
        obs_t = tf.convert_to_tensor(obs_n, dtype=tf.float32)
        obs_t = tf.expand_dims(obs_t, axis=0)

        pre_tanh, logp = self.actor.sample_pre_tanh(obs_t, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None), deterministic=deterministic)
        tanh_a, _ = squash_and_correct(pre_tanh, logp)
        a_env = self.scale_action(tanh_a)
        return a_env.numpy().squeeze(0)

    def update(self):
        if self.replay.size < self.cfg.batch_size:
            return
        obs, acts, rews, next_obs, done = self.replay.sample(self.cfg.batch_size)
        self.compiled_train_step(obs, acts, rews, next_obs, done)

    def _train_step(self, obs, acts, rews, next_obs, done):
        # compute alpha
        if self.log_alpha is not None:
            alpha = tf.exp(self.log_alpha)
            # Original PyTorch code had a clamp, which can help stability.
            alpha = tf.clip_by_value(alpha, 1e-3, 1e10)
        else:
            alpha = self.alpha

        # Critic Update
        with tf.GradientTape() as q_tape:
            # Calculate target Q
            pre_next, logp_next = self.actor.sample_pre_tanh(next_obs, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None), deterministic=False)
            tanh_next, logp_next_corr = squash_and_correct(pre_next, logp_next)
            a_next_env = self.scale_action(tanh_next)
            q1_next = self.q1_targ((next_obs, a_next_env))
            q2_next = self.q2_targ((next_obs, a_next_env))
            q_next = tf.minimum(q1_next, q2_next)
            target_v = q_next - alpha * logp_next_corr
            target_q = rews + (1.0 - done) * self.cfg.gamma * target_v
            target_q = tf.stop_gradient(target_q)

            # Current Q estimates
            q1 = self.q1((obs, acts))
            q2 = self.q2((obs, acts))
            q_loss = tf.reduce_mean(tf.square(q1 - target_q)) + tf.reduce_mean(tf.square(q2 - target_q))

        q_vars = self.q1.trainable_variables + self.q2.trainable_variables
        q_grads = q_tape.gradient(q_loss, q_vars)
        self.q_opt.apply_gradients(zip(q_grads, q_vars))

        # Actor and Alpha update
        with tf.GradientTape(persistent=True) as tape:
            pre_pi, logp_pi = self.actor.sample_pre_tanh(obs, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None), deterministic=False)
            tanh_pi, logp_pi_corr = squash_and_correct(pre_pi, logp_pi)
            a_pi_env = self.scale_action(tanh_pi)
            q1_pi = self.q1((obs, a_pi_env))
            q2_pi = self.q2((obs, a_pi_env))
            q_pi = tf.minimum(q1_pi, q2_pi)

            policy_loss = tf.reduce_mean(alpha * logp_pi_corr - q_pi)

            if self.log_alpha is not None:
                alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(logp_pi_corr + self.target_entropy))

        actor_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        if self.log_alpha is not None:
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_opt.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        del tape

        # Soft updates for target networks
        self._soft_update(self.q1, self.q1_targ)
        self._soft_update(self.q2, self.q2_targ)

    def _soft_update(self, source_net, target_net):
        for s_var, t_var in zip(source_net.trainable_variables, target_net.trainable_variables):
            t_var.assign(self.cfg.tau * s_var + (1.0 - self.cfg.tau) * t_var)

    def train(self):
        env = self.env
        obs, _ = env.reset(seed=self.cfg.seed)
        if self.obs_rstat is not None:
            # Initialize normalization stats with the first observation
            self.normalize_obs(obs.copy())

        ep_ret, ep_len = 0.0, 0
        start = time.time()
        for step in range(1, self.cfg.total_steps + 1):
            self.total_steps = step
            if step < self.cfg.start_steps:
                a = env.action_space.sample()
            else:
                a = self.select_action(obs, deterministic=False)

            next_obs, rew, terminated, truncated, _ = env.step(a)
            done = float(terminated or truncated)
            # store raw obs (not normalized) and action
            self.replay.add(obs.copy(), a.copy(), rew, next_obs.copy(), done)

            obs = next_obs.copy()
            # update obs normalization stats *after* adding to replay buffer
            if self.obs_rstat is not None:
                self.normalize_obs(obs.copy())

            ep_ret += float(rew)
            ep_len += 1

            if done or (ep_len >= env.spec.max_episode_steps):
                # log
                sigma_val = None
                alpha_val = None
                if self.cfg.use_gsde:
                    sigma_val = self.actor.noise.sigma.numpy()
                if self.log_alpha is not None:
                    alpha_val = float(tf.exp(self.log_alpha).numpy())
                print(f"Step {step}, Episode return {ep_ret:.1f}, length {ep_len}, sigma {sigma_val}, alpha {alpha_val}")
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
                if self.obs_rstat is not None:
                    self.normalize_obs(obs.copy())

            # learning step
            if step >= self.cfg.start_steps:
                # do a single SGD update per environment step
                self.update()

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cfg = SACConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    agent = SACAgent(env, cfg)
    agent.train()