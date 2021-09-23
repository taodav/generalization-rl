import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import BaseAgent

from.sarsa_tc import MountainCarTileCoder

criterion = torch.nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.nonlin = nn.ReLU()
        self.i2h = nn.Linear(input_size, input_size//2, bias=False)
        self.h2o = nn.Linear(input_size//2, output_size, bias=False)

    def forward(self, x):
        # 2-layer nn
        x = self.i2h(x)
        x = self.nonlin(x)
        x = self.h2o(x)
        return x


class SarsaAgent(BaseAgent):
    def agent_init(self, agent_init_info):
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"] / agent_init_info["num_tilings"]

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.iht_size = agent_init_info['iht_size']

        self.nn = SimpleNN(self.iht_size, self.num_actions).to(device)
        self.weights_init(self.nn)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
        self.tau = 0.5
        self.updates = 0

        self.tc = MountainCarTileCoder(
            iht_size=self.iht_size,
            num_tilings=agent_init_info['num_tilings'],
            num_tiles=agent_init_info['num_tiles'],
        )

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def get_state_feature(self, state):
        active_tiles = self.tc.get_tiles(*state)
        return torch.FloatTensor([np.eye(self.iht_size)[active_tiles].sum(axis=0)]).to(device)

    def agent_start(self, state):
        state = self.get_state_feature(state)

        with torch.no_grad():
            current_q = self.nn(state)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps = 0
        return action

    def agent_step(self, reward, state):
        # Choose action using epsilon greedy.
        state = self.get_state_feature(state)

        with torch.no_grad():
            current_q = self.nn(state)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        loss = self.batch_train(self.prev_state, self.prev_action, state, action, reward, self.discount)

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1

        return action, loss.item()

    def agent_end(self, reward, state, append_buffer=True):
        state = self.get_state_feature(state)
        if append_buffer:
            self.batch_train(self.prev_state, self.prev_action, state, 0, reward, 0)

    def batch_train(self, state, action, new_state, new_action, reward, discount):
        self.updates += 1
        self.nn.train()
        state_batch = state
        action_batch = torch.LongTensor([action]).view(-1, 1).to(device)
        new_state_batch = new_state
        new_action_batch = torch.LongTensor([new_action]).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor([reward]).to(device)
        discount_batch = torch.FloatTensor([discount]).to(device)

        current_q = self.nn(state_batch)
        sarsa_action_values = current_q.gather(1, action_batch)
        with torch.no_grad():
            new_q = self.nn(new_state_batch)
        sarsa_q = new_q.gather(1, new_action_batch).squeeze_()
        target = reward_batch
        target += discount_batch * sarsa_q
        target = target.view(-1, 1)
        loss = criterion(sarsa_action_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss
