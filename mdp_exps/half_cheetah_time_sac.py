import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gym
import json
import time

import matplotlib.pyplot as plt
import numpy as np

class TimeEmbedding(nn.Module):
    """form the time-embedding"""
    def __init__(self, tdim=50):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / torch.arange(2, tdim + 1, 2).unsqueeze(0)

    def forward(self, t):
        sin_emb = torch.sin(self.freqs.to(t.device) * t)
        cos_emb = torch.cos(self.freqs.to(t.device) * t)
        return torch.cat([sin_emb, cos_emb], dim=-1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std
    
    def sample(self, state, deterministic=False):
        mu, std = self.forward(state)
        if deterministic:
            action = torch.tanh(mu)
            return action * self.max_action, None
        else:       
            normal = Normal(mu, std)
            x = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x)
            log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action * self.max_action, log_prob

# Q-networks
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 (for reducing overestimation bias)
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2

class SAC:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        update_frequency=1,
        min_buffer_size=100,
        batch_size=64,
        buffer_size=10000
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # Copy parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Experience buffer for mini-batch updates
        self.buffer = []
        self.buffer_size = buffer_size
        self.total_steps = 0
        
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1))
        action, _ = self.actor.sample(state, deterministic=deterministic)
        return action.detach().numpy().flatten()
    
    def update_parameters(self):
        if len(self.buffer) < self.min_buffer_size:
            return
            
        # Sample from buffer
        indices = np.random.choice(len(self.buffer), min(self.batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward).reshape(-1, 1))
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done).reshape(-1, 1))
        
        with torch.no_grad():
            # Sample next action and its log probability
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # Target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * (target_q - self.alpha * next_log_prob)
            
        # Current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss calculation
        new_action, log_prob = self.actor.sample(state)
        q1, q2 = self.critic(state, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        self.total_steps += 1
        if self.total_steps % self.update_frequency == 0:
            self.update_parameters()

def custom_reset(env, custom_state):
    # Reset environment and set custom state
    env.reset()  # Optional, some environments require a reset
    env.state = custom_state  # Set the state to a predefined value
    return env.state

def evaluate_policy(agent, current_state, env_name="HalfCheetah-v4", max_future_steps=1000):
    """Compute the cumulative reward of following the current policy in the future from the current state onwards 
    for a fixed number of steps. This is akin to prospective reward.
    """
    eval_env = gym.make(env_name)
    cum_reward = 0.
    state = custom_reset(eval_env, current_state)
    for _ in range(max_future_steps):
        action = agent.select_action(state, deterministic=True)
        state, reward, _, _, _ = eval_env.step(action)
        cum_reward += reward
    return cum_reward

def run_online_learning(env_name="HalfCheetah-v4", max_steps=1000000, eval_freq=5000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize the agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        update_frequency=1,  # Update after each step for truly online learning
        min_buffer_size=100  # Start learning after collecting 100 samples
    )
    
    # Initial state
    state, _ = env.reset()
    episode_reward = 0
    
    # Logging variables
    rewards = []
    avg_speeds = []
    current_speed_window = []
    evaluations = []

    writer = SummaryWriter(log_dir="runs/my_experiment")
    
    for step in range(max_steps):
        # Select action
        action = agent.select_action(state)
        
        # Execute action
        next_state, reward, _, _, _ = env.step(action)
        done = False # because the episode never ends
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Update state and accumulate reward
        state = next_state
        episode_reward += reward
        
        # Track speed (x-velocity is typically the relevant metric for half-cheetah)
        current_speed_window.append(next_state[8])  # Assuming index 8 is x-velocity
        if len(current_speed_window) > 100:
            current_speed_window.pop(0)
        
        # Log performance metrics every 100 steps
        if step % 100 == 0:
            rewards.append(episode_reward)
            avg_speed = np.mean(current_speed_window) if current_speed_window else 0
            avg_speeds.append(avg_speed)
            # print(f"Step: {step}, Avg. Reward: {episode_reward/(step+1):.2f}, Avg. Speed: {avg_speed:.2f}")
        
        # Evaluate policy periodically
        if step % eval_freq == 0:
            future_return = evaluate_policy(agent, state, env_name)
            evaluations.append(future_return)
            print(f"Evaluation at step {step}: Future Return: {future_return:.2f}")
            writer.add_scalar("Future Return", future_return, step)

    writer.close()        
    return rewards, avg_speeds, evaluations, agent

if __name__ == "__main__":
    tic = time.time()
    rewards, speeds, evaluations, agent = run_online_learning(max_steps=5000000)
    toc = time.time()
    print(f"Training time: {(toc - tic)//3600:.2f} hrs")

    torch.save(agent.actor.state_dict(), "online_sac_policy.pth")

    metrics = {
        "rewards": rewards,
        "speeds": speeds,
        "evaluations": evaluations
    }

    with open("online_sac_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)