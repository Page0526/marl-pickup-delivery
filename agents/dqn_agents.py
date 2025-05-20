import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from agents.agent import Agents  # Assuming Agents is in agents.py

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent(Agents):
    def __init__(self, state_dim=6, action_dim=15, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.1, eps_decay=0.995):
        super().__init__()
        self.n_robots = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = 64
        self.steps = 0

    def init_agents(self, state):
        super().init_agents(state)
        self.n_robots = len(state['robots'])

    def encode_state(self, robot_idx, state):
        robot = state['robots'][robot_idx]
        # (x, y, carrying, t)
        return np.array([
            robot[0] / len(self.map), 
            robot[1] / len(self.map[0]),
            robot[2], 
            state['time_step'] / 100.0, 
            self.n_robots / 10.0,
            len(state['packages']) / 20.0
        ], dtype=np.float32)

    def decode_action(self, index):
        moves = ['S', 'L', 'R', 'U', 'D']
        move = moves[index // 3]
        pkg_action = str(index % 3)
        return (move, pkg_action)

    def get_actions(self, state):
        self.state = state
        actions = []
        for i in range(self.n_robots):
            obs = self.encode_state(i, state)
            if np.random.rand() < self.epsilon:
                action_index = np.random.randint(self.action_dim)
            else:
                with torch.no_grad():
                    q_vals = self.q_net(torch.tensor(obs).float().unsqueeze(0))
                    action_index = torch.argmax(q_vals).item()
            actions.append(self.decode_action(action_index))
        return actions

    def remember(self, state, actions, reward, next_state, done):
        for i in range(self.n_robots):
            s = self.encode_state(i, state)
            a = self.encode_action(actions[i])
            s2 = self.encode_state(i, next_state)
            self.buffer.push(s, a, reward, s2, done)

    def encode_action(self, action):
        moves = ['S', 'L', 'R', 'U', 'D']
        move_idx = moves.index(action[0])
        pkg_idx = int(action[1])
        return move_idx * 3 + pkg_idx

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states).float()
        actions = torch.tensor(actions).long().unsqueeze(1)
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float().unsqueeze(1)

        q_vals = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_vals = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_vals = rewards + self.gamma * max_next_q_vals * (1 - dones)

        loss = nn.MSELoss()(q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
