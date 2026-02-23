import gymnasium as gym
from stable_baselines3 import SAC
from CrazyflieFollowEnv import CrazyflieFollowEnv

# 1. Load the Environment 
env = CrazyflieFollowEnv(render_mode="human")

# 2. Load the Trained Brain
model = SAC.load("final_drone_model.zip")

# 3. Enjoy the Show
obs, _ = env.reset()
print("Test started! Press Ctrl+C to stop.")

while True:
   
    action, _states = model.predict(obs, deterministic=True)
    

    obs, reward, done, truncated, info = env.step(action)
    
    
    if done or truncated:
        obs, _ = env.reset()
