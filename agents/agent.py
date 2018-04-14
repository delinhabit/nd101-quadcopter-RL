import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import Adam

from .replay_buffer import ReplayBuffer
from .actor import Actor
from .critic import Critic
from .noise import OUNoise


class DDPGAgent():

    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor models
        self.actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target.model.set_weights(self.actor.model.get_weights())

        # Critic models
        self.critic = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.1
        self.exploration_sigma = 0.3
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99
        self.tau = 0.1

        # Keep track of the score as total_reward / count
        self.total_reward = 0
        self.count = 0
        self.best_score = -np.inf

    def reset_episode(self):
        self.total_reward = 0
        self.count = 0
        self.noise.reset()
        return self.task.reset()

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor.model.predict(state)[0]
        return list(action + self.noise.sample())

    def step(self, state, action, reward, next_state, done):
        self.total_reward += reward
        self.count += 1
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            self.learn(self.memory.sample())

        self.score = self.total_reward / max(self.count, 1)
        if self.score > self.best_score:
            self.best_score = self.score

    def learn(self, experiences):
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(
            self.critic.get_action_gradients([states, actions, 0]),
            (-1, self.action_size),
        )
        self.actor.train([states, action_gradients, 1])

        # Soft-update target models
        self.soft_update(self.critic.model, self.critic_target.model)
        self.soft_update(self.actor.model, self.actor_target.model)

    def soft_update(self, model, target_model):
        weights = np.array(model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
