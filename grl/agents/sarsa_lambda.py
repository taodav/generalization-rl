from enum import IntEnum

import numpy as np

from grl.agents import BaseAgent


class TRACE(IntEnum):
    ACCUMULATE = 1
    REPLACING = 2


class SarsaAgent(BaseAgent):
    def agent_init(self, agent_init_info):
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        self.lam = agent_init_info["lambda"]
        # default trace type to accumulate
        self.trace_type = agent_init_info.get("trace_type", TRACE.ACCUMULATE)
        # allocate a memory for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.


    def agent_start(self, state):
        self.z = np.zeros((self.num_states, self.num_actions))
        state = state[0]
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        self.steps = 0
        return action

    def agent_step(self, reward, state):
        state = state[0]
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        delta = (reward + self.discount * self.q[state, action] - self.q[self.prev_state, self.prev_action])
        self.z *= self.discount * self.lam
        if self.trace_type == TRACE.ACCUMULATE:
            self.z[self.prev_state, self.prev_action] += 1
        else:
            self.z[self.prev_state, self.prev_action] = 1
        self.q += self.step_size * delta * self.z
        self.prev_state = state
        self.prev_action = action
        self.steps += 1
        return action

    def agent_end(self, reward, state, append_buffer=True):
        delta = self.step_size * (reward - self.q[self.prev_state, self.prev_action])
        self.z *= self.discount * self.lam
        if self.trace_type == TRACE.ACCUMULATE:
            self.z[self.prev_state, self.prev_action] += 1
        else:
            self.z[self.prev_state, self.prev_action] = 1
        self.q += self.step_size * delta * self.z
