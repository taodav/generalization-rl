import numpy as np
import torch
import copy

from .agent import BaseAgent
from grl.agents.sarsa_nn import Transition, device, SimpleNN
from grl.replay import Replay

criterion = torch.nn.MSELoss()


class DQNAgent(BaseAgent):

    def agent_init(self, agent_init_info):
        # Store the parameters provided in agent_init_info.
        self.batch_size = agent_init_info["batch_size"]
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]

        self.use_replay = agent_init_info.get("use_replay", True)
        self.use_target = agent_init_info.get("use_target", True)

        self.max_replay_size = agent_init_info["max_replay_size"]
        self.update_target_interval = agent_init_info["update_target_interval"]

        self.init_steps = agent_init_info.get("init_steps", 1000)
        self.updates_per_step = agent_init_info.get("updates_per_step", 1)

        if not self.use_replay:
            self.updates_per_step = 1

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.replay = Replay(max_size=self.max_replay_size,
                             im_size=(2, ),
                             rand_generator=self.rand_generator,
                             stack=1)
        self.nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.weights_init(self.nn)

        if self.use_target:
            self.target_nn = copy.deepcopy(self.nn).requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
        self.tau = 0.5
        self.updates = 0
        # This is OVERALL steps since init, not episode steps.
        self.steps = 0

        self.prev_action_value = None
        self.prev_state = None
        self.prev_action = None

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def update_target(self):
        self.target_nn.load_state_dict(self.nn.state_dict())

    def get_state_feature(self, state):
        return torch.FloatTensor(state)

    def agent_start(self, state):
        self.replay.start(state)

        state = self.get_state_feature(state)
        # Choose action using epsilon greedy.
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
        return action

    def agent_step(self, reward, state):
        self.replay.put(self.prev_action, reward, False, state)

        # Choose action using epsilon greedy.
        state = self.get_state_feature(state)

        with torch.no_grad():
            current_q = self.nn(state)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        loss = torch.tensor(0).to(torch.float)
        if self.steps > self.init_steps:
            for u in range(self.updates_per_step):
                if self.use_replay:
                    s, a, r, ns, done = self.replay.sample(self.batch_size)
                else:
                    s, a, r, ns, done = self.prev_state.unsqueeze(0), \
                                        np.array([self.prev_action]),\
                                        np.array([reward]), \
                                        state.unsqueeze(0), \
                                        np.array([False])
                loss += self.batch_train(s, a, ns, r, np.ones_like(r) * self.discount)

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1

        return action, loss.item()

    def agent_end(self, reward, state, append_buffer=True):
        self.replay.put(self.prev_action, reward, True, state)
        loss = torch.tensor(0).to(torch.float)
        if append_buffer:
            for u in range(self.updates_per_step):
                if self.use_replay:
                    s, a, r, ns, done = self.replay.sample(self.batch_size)
                else:
                    state = self.get_state_feature(state)
                    s, a, r, ns, done = self.prev_state.unsqueeze(0), \
                                        np.array([self.prev_action]), \
                                        np.array([reward]), \
                                        state.unsqueeze(0), \
                                        np.array([False])
                loss += self.batch_train(s, a, ns, r, np.ones_like(r) * self.discount)
        return loss.item()

    def batch_train(self, state, action, new_state, reward, discount):
        self.updates += 1
        self.nn.train()
        state_batch = self.get_state_feature(state).to(device)
        action_batch = torch.LongTensor(action).unsqueeze(-1).to(device)
        new_state_batch = self.get_state_feature(new_state).to(device)
        reward_batch = torch.FloatTensor(reward).to(device)
        discount_batch = torch.FloatTensor(discount).to(device)

        current_q = self.nn(state_batch)
        q_learning_action_values = current_q.gather(1, action_batch)
        with torch.no_grad():
            if self.use_target:
                new_q = self.target_nn(new_state_batch)
            else:
                new_q = self.nn(new_state_batch)

        # max_q = new_q.gather(1, new_action_batch).squeeze_()
        max_q = new_q.max(dim=-1).values
        target = reward_batch
        target += discount_batch * max_q
        target = target.view(-1, 1)
        loss = criterion(q_learning_action_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss
