import numpy as np
from physics_sim import PhysicsSim


class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent.
    """

    def __init__(self, init_state, target_state, runtime=5.0, action_repeats=3):
        """
        Initialize a Task object.

        The initial and target states should be iterables with 12 float items
        that represent:
          - pose (x, y, z, phi, theta, psi)
          - velocities (x_v, y_v, z_v)
          - angle velocities (phi_v, theta_v, psi_v)

        :param tuple init_state: The initial state of the quadcopter
        :param tuple target_state: The target state of the quadcopter
        :param float runtime: Time limit for each episode
        :param int action_repeats: Number of times to repeat an action
        """
        assert len(init_state) == 12
        assert len(target_state) == 12

        # Initialize the state arrays
        self.current_state = np.array(init_state, dtype=np.float64)
        self.target_state = np.array(target_state, dtype=np.float64)
        self.action_repeats = action_repeats

        # We're sharing current_state with the simulator so that we don't
        # need to constantly copy the data from the simulator into the current
        # state
        self.sim = PhysicsSim(
            self.current_state[:6],
            self.current_state[6:9],
            self.current_state[9:],
            runtime
        )

        self.current_state_size = len(self.current_state)
        self.state_size = self.current_state_size * self.action_repeats
        self.action_size = 4
        self.action_low = 0
        self.action_high = 900

    def reset(self):
        """
        Reset the sim to start a new episode.
        """
        self.sim.reset()
        return np.repeat(self.current_state, self.action_repeats)

    def get_reward(self):
        """
        Use current state to compute the reward.
        """
        error = abs(self.current_state - self.target_state)
        return 1.0 - 0.3 * error[:3].sum()

    def step(self, rotor_speeds):
        """
        Use action to obtain next state, reward, and done.
        """
        reward = 0
        all_states = np.zeros(self.state_size)

        for i in range(0, self.state_size, self.current_state_size):
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            all_states[i:i + self.current_state_size] = self.current_state

        return all_states, reward, done
