import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

from pandas._typing import F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Определение архитектуры сети
class DQNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_actions):
        super(DQNLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.init_hidden(x)

        x, hidden_state = self.lstm(x, hidden_state)
        x = x[:, -1, :]
        return self.linear(x), hidden_state

    def init_hidden(self, x):
        batch_size = x.size(0)
        hidden_size = next(self.parameters()).size(-1)
        hidden = (torch.zeros(1, batch_size, hidden_size).to(x.device),
                  torch.zeros(1, batch_size, hidden_size).to(x.device))
        return hidden


# Определение агента DQN
class DQNAgent:
    def __init__(self, input_dim, hidden_size, num_actions, lr, gamma, epsilon, epsilon_decay, min_epsilon,
                 memory_size):
        self.model = DQNLSTM(input_dim, hidden_size, num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=memory_size)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def select_action(self, state, hidden_state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values, hidden_state = self.model(state_tensor, hidden_state)
                action = torch.argmax(q_values).item()
        return action, hidden_state

    def store_transition(self, state, action, next_state, reward, done):
        if state is None or next_state is None:
            return

        state = torch.tensor(state, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        action = torch.tensor([action], dtype=torch.int64).to(device)
        reward = torch.tensor([reward], dtype=torch.float32).to(device)

        self.memory.append(self.Transition(state, action, next_state, reward, done))

    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = random.sample(self.memory, batch_size)
        batch = self.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        q_values, _ = self.model(state_batch)
        state_action_values = q_values.gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        next_q_values, _ = self.model(non_final_next_states)
        next_state_values[non_final_mask] = next_q_values.max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
