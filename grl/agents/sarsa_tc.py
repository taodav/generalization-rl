import grl.tiles3 as tc
import numpy as np
from grl.agents import BaseAgent

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07


# SARSA
class SarsaAgent(BaseAgent):

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
        }

        """
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]

        # We need to divide by number of tilings
        # in order to prevent divergence.
        self.step_size = agent_init_info["step_size"] / agent_init_info['num_tilings']
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.w = np.ones((self.num_actions, agent_init_info['iht_size']))

        self.tc = MountainCarTileCoder(
            iht_size=agent_init_info['iht_size'],
            num_tilings=agent_init_info['num_tilings'],
            num_tiles=agent_init_info['num_tiles'],
            position_min=agent_init_info['position_min'],
            position_max=agent_init_info['position_max'],
            velocity_min=agent_init_info['velocity_min'],
            velocity_max=agent_init_info['velocity_max']
        )

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
        self.w[self.last_action][self.previous_tiles] += self.step_size * td_error

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
        if append_buffer:
            self.w[self.last_action][self.previous_tiles] += self.step_size * td_error
        return td_error


class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8,
                 position_min=-1.2, position_max=0.6,
                 velocity_min=-0.07, velocity_max=0.07):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

        self.position_min = position_min
        self.position_max = position_max
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max

        self.position_scale = self.num_tiles / (self.position_max - self.position_min)
        self.velocity_scale = self.num_tiles / (self.velocity_max - self.velocity_min)

    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.

        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """
        # Set the max and min of position and velocity to scale the input
        # position_scale = self.num_tiles / (POSITION_MAX - POSITION_MIN)
        # velocity_scale = self.num_tiles / (VELOCITY_MAX - VELOCITY_MIN)

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        tiles = tc.tiles(self.iht, self.num_tilings, [position * self.position_scale,
                                                      velocity * self.velocity_scale])

        return np.array(tiles)
