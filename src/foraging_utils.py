# import dependencies
import numpy as np
from gymnasium import register
import random

'''
an optimal agent should switch patch so that it can arrive at
the next rewarching patch just when it gets refreshed
''' 
def optimal_linear(
        map_name = "short",
        base_reward = 10.0,
        decay_rate = 0.6,
        reward_period = 10,
        session_duration = 100
):

    STAY = 0
    TRAVEL = 1

    if map_name == "short":
        travel_cost = 4
    elif map_name == "long":
        travel_cost = 6
    else:
        print("Customized map is not available")
        return None
    
    reward_per_step = np.zeros((session_duration,1))
    action_per_step = np.zeros((session_duration,1))
    action = None
    for t in range(0, session_duration):
        if action != TRAVEL or t ==1:
            action_per_step[t] = STAY
            travel_step = 0
            # pdb.set_trace()
            # which patch available
            cur_avail_patch_id = t//reward_period

            # this patch refresh time
            patch_start = reward_period * cur_avail_patch_id
            
            # how many steps in this patch already
            steps_cur_patch = t - patch_start
            reward_per_step[t] = base_reward * decay_rate**(steps_cur_patch)

            # optimal action (stay or travel)
            if steps_cur_patch + travel_cost < reward_period:
                action = STAY
            elif steps_cur_patch + travel_cost >= reward_period:
                action = TRAVEL
        else: # action == travel
            action_per_step[t] = TRAVEL
            reward_per_step[t] = 0.0
            travel_step += 1
            if travel_step ==travel_cost:
                # next action should be stay
                action = STAY
                
                # if arriving at the rewarding patch, then reward should be nonzero (even though the agent is travling)
                cur_avail_patch_id = t//reward_period
                # this patch refresh time
                patch_start = reward_period * cur_avail_patch_id
                
                # how many steps in this patch already
                steps_cur_patch = t - patch_start
                reward_per_step[t] = base_reward * decay_rate**(steps_cur_patch)
                # print(reward_per_step[t]) #should be 10

    return reward_per_step, action_per_step


def chance_linear(
        env,
        num_reps = 100000
):
    

    reward_total_rep = []
    action_total_rep = []
    for rep in range(num_reps):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = env.reset()
        done = False

        reward_per_rep = []
        action_per_rep = []
        while not done:
            action = random.choice([0, 1, 2]) # randomly choose an actions
            action_per_rep.append(action)
            location, reward, terminated, truncated, info = env.step(action)
            reward_per_rep.append(reward)
            done = terminated or truncated 

        reward_total_rep.append(reward_per_rep)
        action_total_rep.append(action_per_rep)
        
        reward_per_rep = []
        action_per_rep = []

    return reward_total_rep, action_total_rep



'''
Prepare input for the algorithm, pad with 0s when trial length is not enough
reward history, action history, position history
params: reward_deque:  a deque of past reward
        reward_length: trials of reward history known to agent
        action_deque: a deque of past actions
        action_length
        location_deque: a deque of past positions
        position_length
'''
def input_function(reward_deque, action_deque, location_deque):
    reward_length = reward_deque.maxlen
    action_length = action_deque.maxlen
    position_length = location_deque.maxlen

    reward_array = np.array(reward_deque)
    action_array = np.array(action_deque)
    position_array = np.array(location_deque)

    if len(reward_array) < reward_length:
        # pad the reward array with 0s
        reward_array = np.pad(reward_array, pad_width=(0, reward_length - len(reward_array)), mode='constant', constant_values=0)

    if len(action_array) < action_length:
        # pad the action array with 0s
        action_array = np.pad(action_array, pad_width=(0, action_length - len(action_array)), mode='constant', constant_values=0)

    if len(position_array) < position_length:
        # pad the reward array with 0s
        position_array = np.pad(position_array, pad_width=(0, position_length - len(position_array)), mode='constant', constant_values=0)


    input = np.concatenate((reward_array,action_array, position_array))
    return input.reshape(1, -1)


def register_custom_env(
        env_name = 'ForagingPlaygroundLinear-v0', 
        map_name = 'short',
        base_reward = 10.0,
        decay_rate = 0.6,
        reward_period = 10,
        session_duration = 100
        ):
    
    if env_name == 'ForagingPlaygroundLinear-v0':
        entry_point = 'linear_foraging:foraging_playground_linear'
    elif env_name == 'ForagingPlaygroundGrid-v0':
        entry_point = 'grid_foraging:foraging_playground_grid'
    
    register(id = env_name,
             entry_point = entry_point, 
             kwargs = {
                'render_mode': 'rgb_array', #'human' is not suitable for training
                'map_name': map_name,
                'base_reward': base_reward,
                'decay_rate': decay_rate,
                'reward_period': reward_period,
                'session_duration': session_duration,
             },
             max_episode_steps = session_duration)

