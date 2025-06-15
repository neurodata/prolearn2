import gymnasium as gym
from linear_foraging import foraging_playground_linear  
import pygame

'''
Play the foraging game yourself!

If you want to play the game in text, 
        -> choose 'rgb_array'
If you want to play the game in animation, and use keyboard to control the agent
        -> choose 'human'
'''
render_mode = "human"

# Initialize the environment
gym.register(id = 'ForagingPlaygroundLinear-v0',
             entry_point = foraging_playground_linear, 
             kwargs = {
                'render_mode': render_mode,
                'map_name': "short",
                'base_reward': 10.0,
                'decay_rate': 0.6,
                'reward_period': 10,
                'session_duration': 100,
             },
             max_episode_steps = 100)
env = gym.make('ForagingPlaygroundLinear-v0')

state, info = env.reset()

# Mapping user input to actions
STAY = 0
LEFT = 1
RIGHT = 2

key_to_action = {
    pygame.K_SPACE: STAY,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
    pygame.K_e: "exit"
}
action_mapping = {
    'stay': STAY,
    'left': LEFT,
    'right': RIGHT
}
def get_action_from_key(event):
    return key_to_action.get(event.key, None)


if render_mode == "human":
    print("Choose your actions (stay, left, right) by pressing the corresponding keys (space, left, right). Press 'E' to quit.")
    pygame.init()
    done = False
    total_energy = 0
    time = 0
    while not done:
        env.render()
        # PRESS KEY CONTROL VERSION
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                action = get_action_from_key(event)
                if action == "exit":
                    done = True
                    print("Game Over!")
                    print(f"Congratulations! You have earned {total_energy} reward in {time} timesteps)")
                    break
                elif action is not None:
                    state, reward, done, truncated, info  = env.step(action)
                    total_energy = info["total_reward"]
                    time = info["time"]
                    print(f"State: {state}, Reward: {reward}, Total Reward: {total_energy}, Time: {time}")
                    if done:
                        print("Game Over!")
                        print(f"Congratulations! You have earned {total_energy} reward in {time} timesteps)")
                        state, info = env.reset()
                        break
else:
    print("Enter your actions (stay, left, right). Type 'exit' to quit.")
    pygame.init()
    done = False
    total_energy = 0
    time = 0
    while not done:
        env.render()
        action = input("Action: ").strip().lower()
        if action == 'exit':
            done = True
            print("Game Over!")
            print(f"Congratulations! You have earned {total_energy} reward in {time} timesteps)")
            break
        if action in action_mapping:
            action_code = action_mapping[action]
            state, reward, done, truncated, info = env.step(action_code)
            total_energy = info["total_reward"]
            time = info["time"]
            print(f"State: {state}, Reward: {reward}, Total Reward: {total_energy}, Time: {time}")
        else:
            print("Invalid action. Please enter one of 'stay', 'left', 'right'.")

        if done:
            print("Game Over!")
            print(f"Congratulations! You have earned {total_energy} reward in {time} timesteps)")
            state, info = env.reset()
            break
        
# Close the environment
env.close()
pygame.quit()
