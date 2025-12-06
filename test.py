import gymnasium as gym
from stable_baselines3 import SAC
from CrazyflieFollowEnv import CrazyflieFollowEnv

# 1. Load the Environment (With Visuals ON this time!)
env = CrazyflieFollowEnv(render_mode="human")

# 2. Load the Trained Brain
# Make sure the filename matches what you saved!
model = SAC.load("final_drone_model.zip")

# 3. Enjoy the Show
obs, _ = env.reset()
print("Test started! Press Ctrl+C to stop.")

while True:
    # Ask the AI what to do
    action, _states = model.predict(obs, deterministic=True)
    
    # Do it
    obs, reward, done, truncated, info = env.step(action)
    
    # If the drone crashes or time runs out, reset
    if done or truncated:
        obs, _ = env.reset()