import numpy as np
from grl.agents import BaseAgent

from .sarsa_tc import MountainCarTileCoder
from .sarsa_lambda import TRACE

class SarsaLambdaTCAgent(BaseAgent):

    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
            iht_size (int): index hash table size,
            num_tilings (int): number of tilings to use,
            num_tiles (int): number of tiles (over one dimension) in a tiling
            lambda (int): lambda parameter for eligibility trace
        }

        """
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.trace_type = agent_init_info.get("trace_type", TRACE.ACCUMULATE)
        self.lam = agent_init_info["lambda"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.w = np.ones((self.num_actions, agent_init_info['iht_size']))
        self.z = np.zeros((self.num_actions, agent_init_info['iht_size']))

        self.tc = MountainCarTileCoder(iht_size=agent_init_info['iht_size'],
                                       num_tilings=agent_init_info['num_tilings'],
                                       num_tiles=agent_init_info['num_tiles'])

    def select_action(self, tiles):
        """
        Selects an action using epsilon greedy
        Args:
        tiles - np.array, an array of active tiles
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """
        action_values = []
        chosen_action = None

        for action in self.w:
            action_values.append(action[tiles].sum())

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)
        else:
            chosen_action = self.argmax(action_values)

        return chosen_action, action_values[chosen_action]

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.feature = None
        position, velocity = state

        active_tiles = self.tc.get_tiles(position, velocity)
        current_action, _ = self.select_action(active_tiles)

        self.z = np.zeros_like(self.z)
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        self.steps = 0
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        position, velocity = state

        active_tiles = self.tc.get_tiles(position, velocity)
        current_action, action_value = self.select_action(active_tiles)

        td_target = reward + self.discount * action_value
        td_error = td_target - self.w[self.last_action][self.previous_tiles].sum()
        self.z *= self.discount * self.lam
        if self.trace_type == TRACE.ACCUMULATE:
            self.z[self.last_action, active_tiles] += 1
        else:
            self.z[self.last_action, active_tiles] = 1
        self.w[self.last_action][self.previous_tiles] += self.step_size * td_error * self.z

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        self.steps += 1
        return self.last_action, td_error

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        td_target = reward
        td_error = td_target - self.w[self.last_action][self.previous_tiles].sum()

        self.z *= self.discount * self.lam
        if self.trace_type == TRACE.ACCUMULATE:
            self.z[self.last_action, self.previous_tiles] += 1
        else:
            self.z[self.last_action, self.previous_tiles] = 1
        self.w[self.last_action][self.previous_tiles] += self.step_size * td_error * self.z
        return td_error
