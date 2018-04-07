import random
from collections import deque

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from .base import AgentBase


class DQNAgent(AgentBase):

    def __init__(self, task, batch_size=32):
        super().__init__(task)

        self.memory = deque(maxlen=1000)
        self.batch_size = batch_size
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

        self.score = -np.inf
        self.best_score = -np.inf
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        # Obligatory exploration
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(self.action_low, self.action_high, size=self.action_size)

        # Choose the action with the highest value as predicted
        # by our model.
        return self.predict(state)

    def predict(self, state):
        return self.model.predict(np.reshape(state, [1, self.state_size]))

    def step(self, *args):
        self.memory.append(args)
        super().step(*args)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # TODO
            pass

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.score = self.total_reward / max(self.count, 1)
        if self.score > self.best_score:
            self.best_score = self.score
