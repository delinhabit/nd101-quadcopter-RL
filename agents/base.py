import abc


class AgentBase(abc.ABC):

    def __init__(self, task):
        """
        Initialize the agent with the provided task (environment).
        """
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.reset_episode()

    def reset_episode(self):
        """
        Reset to a new episode.
        """
        self.total_reward = 0.0
        self.count = 0
        return self.task.reset()

    @abc.abstractmethod
    def act(self, state):
        """
        Choose an action based on the given state and current policy.
        """

    def step(self, state, action, reward, next_state, done):
        """
        Incorporate the experience gained by interacting with the environment.
        """
        self.total_reward += reward
        self.count += 1

        if done:
            self.learn()

    @abc.abstractmethod
    def learn(self):
        """
        Learn from the previous experience.
        """
