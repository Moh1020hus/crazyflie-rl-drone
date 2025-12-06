import gymnasium as gym
import os # 1. Import os to count your CPU cores
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env # 2. Import vectorizer tool
from stable_baselines3.common.vec_env import SubprocVecEnv # 3. Import Multiprocessing

from CrazyflieFollowEnv import CrazyflieFollowEnv

def main():
    # --- CHANGE 1: DETERMINE CPU CORES ---
    # Count how many logical cores your PC has (e.g., 8, 12, 16)
    num_cpu = os.cpu_count()
    print(f"Detected {num_cpu} CPU cores. Creating {num_cpu} parallel environments...")

    # --- CHANGE 2: CREATE PARALLEL ENVIRONMENTS ---
    # "SubprocVecEnv" runs each environment in a separate process (True Parallelism)
    # We set render_mode=None because you can't open 12 windows at once!
    env = make_vec_env(
        CrazyflieFollowEnv, 
        n_envs=num_cpu, 
        seed=0, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": None} 
    )
    
    # --- CHANGE 3: SETUP MODEL ON CPU ---
    # device="cpu": Moves the Neural Network math to the CPU
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,
        tensorboard_log="./sac_drone_tensorboard/",
        device="cpu"  # <--- Forced CPU usage
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='sac_drone')

    print("-----------------------------------------")
    print("STARTING PARALLEL TRAINING... Press Ctrl+C to stop")
    print("-----------------------------------------")

    # Since we are running N environments, 1 step here = N steps of experience.
    # So 10,000 "steps" in the learner might actually be 10,000 * num_cpu total frames.
    model.learn(total_timesteps=20000, callback=checkpoint_callback)
    
    model.save("final_drone_model")
    print("Training finished. Model saved!")
    
    env.close()

if __name__ == "__main__":
    # IMPORTANT: On Windows, multiprocessing MUST be inside this main block
    main()