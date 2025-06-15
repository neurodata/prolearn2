from __future__ import annotations
from foraging_utils import input_function, register_custom_env
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import gymnasium as gym
import pickle
import argparse

'''
Architecture:
Feedforward network (1 hidden layer)
Two output head: actor (action evaluation), and critic (policy evaluation)
'''
class retrospective_forager(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int
        ):
        super(retrospective_forager, self).__init__()
        
        self.device = device
        self.n_envs = n_envs

        # Shared hidden layer
        self.shared_fc = nn.Linear(n_features, 32).to(self.device) 
        
        # actor and critic
        self.actor = nn.Linear(32, n_actions).to(self.device)
        self.critic = nn.Linear(32, 1).to(self.device)

        # Initialize weights
        nn.init.kaiming_normal_(self.shared_fc.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.xavier_uniform_(self.critic.weight)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def forward(self, x):
        x = x.to(self.device,  dtype=torch.float32) if isinstance(x, torch.Tensor) else torch.tensor(x, device=self.device, dtype=torch.float32)
        x = torch.relu(self.shared_fc(x))
        
        # Policy (actor) output 
        action_logits = self.actor(x)
        
        # Value (critic) output
        state_values = self.critic(x)
        
        return action_logits, state_values
    
    def select_action(self,x):
        action_logits, state_values = self.forward(x)

        action_pd = torch.distributions.Categorical(logits=action_logits) 
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy
    
    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
    ):
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=self.device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = rewards[t] +  gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        critic_loss = advantages.pow(2).mean()

        actor_loss = (-(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean())

        total_loss = actor_loss + critic_loss
        return total_loss
    
    def get_future_reward(
        self,
        env,
        reward_deque,
        action_deque,
        location_deque,
        horizon
    ):
        ''' 
        Estimate the future loss by projecting into the future using the current policy.
        '''
        
        saved_state = env.unwrapped.get_state() # state of the environment
        # input for the agent
        states = input_function(reward_deque, action_deque, location_deque)
        
        ep_rewards = np.zeros((10,horizon))
        for iteration in range(10):
            # from here, sample the future trajectory using the current policy
            reward_deque_trajectory = reward_deque
            action_deque_trajectory = action_deque
            location_deque_trajectory = location_deque
            for step in range(horizon):
                
                actions, _, _, _ = self.select_action(states)

                location, rewards, terminated, _, _ = env.step(
                    actions.cpu().numpy()
                )

                reward_deque_trajectory.extend([rewards])
                action_deque_trajectory.extend([actions.cpu().numpy()[0]])
                location_deque_trajectory.extend([location])
                
                states = input_function(reward_deque, action_deque, location_deque)
                ep_rewards[iteration][step] = rewards
                if terminated:
                    break
            # keep the env unchanged
            env.unwrapped.return_state(saved_state)
        
        mean_ep_reward = np.mean(ep_rewards, axis = 0)
        se_ep_reward = np.std(ep_rewards, axis=0) / np.sqrt(ep_rewards.shape[0])

        return mean_ep_reward, se_ep_reward

    
    def update_parameters(
        self, total_loss: torch.Tensor, max_grad_norm: float = 0.5
    ):
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        total_loss.backward()

        # add gradient clipping for more stable behavior
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)

        self.actor_optim.step()
        self.critic_optim.step()


def main(session_duration,
         n_episode,
         n_steps_per_update,
         n_reps, # number of repetitions, could be set from 1 to any integer
         gamma,
         lam,  # hyperparameter for GAEent_coef = 0.01  
         ent_coef, # coefficient for the entropy bonus (to encourage exploration)
         actor_lr,
         critic_lr,
         history_length # length of past trial's history as part of the model input
        ): 
    # Set up the environment
    env_name = 'ForagingPlaygroundLinear-v0'
    register_custom_env(
        env_name = env_name, 
        map_name = 'short',
        base_reward = 10.0,
        decay_rate = 0.6,
        reward_period = 10,
        session_duration = session_duration
    )
    env = gym.make(env_name)

    # Explicitly define single observation and action spaces
    env.single_observation_space = gym.spaces.Discrete(7)
    env.single_action_space = gym.spaces.Discrete(3)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)

    '''some params'''
    # environment hyperparams
    n_env = 1
    n_updates = n_episode * session_duration//n_steps_per_update

    # observation space (input)
    obs_shape = 3 * history_length # reward_length + action_length + location_length 
    action_shape = env.single_action_space.n
    
    # Start running the algorithm
    reward_total_rep = []
    loss_total_rep = []
    entropies_total_rep = []
    action_total_rep = []
    prospective_reward_total_rep = []
    prospective_se_total_rep = []

    for n_rep in tqdm(range(n_reps)):
        # init the agent
        agent = retrospective_forager(obs_shape, action_shape, device, critic_lr, actor_lr,  n_env)
        
        network_losses = []
        prospective_reward = []
        prospective_se = []

        # input function (agent observation space)
        reward_deque = deque([], maxlen=history_length)
        location_deque = deque([], maxlen=history_length)
        action_deque = deque([], maxlen=history_length)

        # record reward per episode for all episodes
        reward_total_episode = []
        reward_per_episode = []
        actions_per_episode = []
        actions_total_episode = []
        entropy_per_episode = []

        count = 0 # keep a record of how many updates are done
        project_horizon = n_steps_per_update

        for sample_phase in tqdm(range(n_updates)):
            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(n_steps_per_update, n_env, device=device)
            ep_rewards = torch.zeros(n_steps_per_update, n_env, device=device)
            ep_action_log_probs = torch.zeros(n_steps_per_update, n_env, device=device)
            ep_entropy = torch.zeros(n_steps_per_update, n_env, device=device)
            masks = torch.zeros(n_steps_per_update, n_env, device=device)

            
            # at the start of training reset all env to get an initial state
            if sample_phase == 0:
                location, info = env.reset()
                location_deque.extend(np.array([location]))
                states = input_function(reward_deque, action_deque, location_deque)


            # play n steps in our parallel environments to collect data
            for step in range(n_steps_per_update):
                # select an action A_{t} using S_{t} as input for the agent
                actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                    states
                )

                actions_per_episode.append(actions.cpu().numpy())
                entropy_per_episode.append(entropy.detach().mean().cpu().numpy())

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                location, rewards, terminated, _, _ = env.step(
                    actions.cpu().numpy()
                )
                reward_per_episode.append(rewards)

                reward_deque.extend([rewards])
                action_deque.extend([actions.cpu().numpy()[0]])
                location_deque.extend([location])
                states =  input_function(reward_deque, action_deque, location_deque)

                ep_value_preds[step] = torch.squeeze(state_value_preds)
                ep_rewards[step] = torch.tensor(rewards, device=device)
                ep_action_log_probs[step] = action_log_probs
                ep_entropy[step] = entropy

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor(0 if terminated else 1, dtype=torch.float32)
                
                if terminated:
                    reward_total_episode.append(reward_per_episode)
                    actions_total_episode.append(actions_per_episode)
                    reward_per_episode = []
                    actions_per_episode = []
                    count += 1
                    location, info = env.reset()
                    location_deque.extend(np.array([location]))
                    states = input_function(reward_deque,action_deque, location_deque) 
                    # need to refresh these deques between episode?

            # calculate the losses for actor and critic
            total_loss = agent.get_losses(
                ep_rewards,
                ep_action_log_probs,
                ep_value_preds,
                ep_entropy,
                masks,
                gamma,
                lam,
                ent_coef,
            )

            # estimate the future loss 
            future_mean_rewards, future_se_rewards = agent.get_future_reward(
                env,
                reward_deque, 
                action_deque, 
                location_deque,
                project_horizon,
            )

            # update the actor and critic networks
            agent.update_parameters(total_loss)

            # log the losses and entropy
            network_losses.append(total_loss.detach().cpu().numpy())
            prospective_reward.append(future_mean_rewards)
            prospective_se.append(future_se_rewards)

        print(f"episode {count} done")
        reward_total_rep.append(reward_total_episode)
        action_total_rep.append(actions_total_episode)
        prospective_reward_total_rep.append(prospective_reward)
        prospective_se_total_rep.append(prospective_se)
        loss_total_rep.append(network_losses)
        entropies_total_rep.append(entropy_per_episode)
        
    
    with open(f'/results/retrospective_forager.pkl', 'wb') as f:
        pickle.dump((reward_total_rep, action_total_rep, loss_total_rep, entropies_total_rep, prospective_reward_total_rep, prospective_se_total_rep), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a reinforcement learning experiment.")

    # Add arguments with default values
    parser.add_argument("--session_duration", type=int, default=500000, help="Total step of each session")
    parser.add_argument("--n_episode", type=int, default=1, help="Number of episodes")
    parser.add_argument("--n_steps_per_update", type=int, default=40, help="Number of steps per policy update")
    parser.add_argument("--n_reps", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor (gamma)")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--actor_lr", type=float, default=0.001, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=0.001, help="Critic learning rate")
    parser.add_argument("--history_length", type=int, default=20, help="History length")

    # Parse arguments
    args = parser.parse_args()

    # Pass the parsed arguments to the main function
    main(
        session_duration=args.session_duration,
        n_episode=args.n_episode,
        n_steps_per_update=args.n_steps_per_update,
        n_reps=args.n_reps,
        gamma=args.gamma,
        lam=args.lam,
        ent_coef=args.ent_coef,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        history_length=args.history_length,
    )