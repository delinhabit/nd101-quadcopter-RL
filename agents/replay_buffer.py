import random
from collections import namedtuple
from collections import deque


Experience = namedtuple('Experience', [
    'state',
    'action',
    'reward',
    'next_state',
    'done',
])


class ReplayBuffer():
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
