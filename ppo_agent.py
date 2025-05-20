import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = []

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs)
        logits, _ = self.model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def store(self, obs, action, reward, next_obs, done, log_prob):
        self.memory.append((obs, action, reward, next_obs, done, log_prob))

    def train(self):
        if not self.memory:
            return

        obs, actions, rewards, next_obs, dones, old_log_probs = zip(*self.memory)
        self.memory.clear()

        obs = torch.FloatTensor(obs)
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(old_log_probs)
        returns = []

        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        logits, values = self.model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = values.squeeze()

        advantages = returns - values.detach()
        ratio = (log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - values).pow(2).mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
