import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

class CrazyflieFollowEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(CrazyflieFollowEnv, self).__init__()
        self.render_mode = render_mode
        
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
       
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        
        self.state = None
        self.target_area = 0.1 
        self.steps_left = 0
        
        
        self.person_world_pos = np.array([0.0, 0.0, 1.0]) 
        self.person_velocity = np.array([0.0, 0.0])
        self.obstacle_ids = [] 
        
        self.physics_client = None
        self.drone_id = None
        self.face_id = None
        
       
        if self.render_mode == "human":
            connection_mode = p.GUI    
        else:
            connection_mode = p.DIRECT 
            
      
        self.physics_client = p.connect(connection_mode)
        
      
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
       
        face_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 0.8, 0.6, 1])
        self.face_id = p.createMultiBody(baseVisualShapeIndex=face_visual, basePosition=self.person_world_pos)

        
        try:
            self.drone_id = p.loadURDF("quadrotor.urdf", [0, -0.5, 1], globalScaling=0.5)
        except:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02], rgbaColor=[0, 0, 1, 1])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02])
            self.drone_id = p.createMultiBody(baseVisualShapeIndex=vis, baseCollisionShapeIndex=col, basePosition=[0, -0.5, 1])
        
       
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=90, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])
       
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
       
        self.person_world_pos = np.array([0.0, 0.0, 1.0])
        self.person_velocity = np.random.uniform(-0.01, 0.01, size=2)
        
        for obs in self.obstacle_ids:
            p.removeBody(obs)
        self.obstacle_ids = []
        
        for _ in range(5):
            x_pos = np.random.choice([-1, 1]) * np.random.uniform(1.0, 2.5)
            y_pos = np.random.choice([-1, 1]) * np.random.uniform(1.0, 2.5)
            
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5], rgbaColor=[0.4, 0.4, 0.4, 1])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.5])
            obs_id = p.createMultiBody(baseVisualShapeIndex=vis, baseCollisionShapeIndex=col, basePosition=[x_pos, y_pos, 0.5])
            self.obstacle_ids.append(obs_id)

        random_x = np.random.uniform(-0.5, 0.5)
        random_y = np.random.uniform(-0.5, 0.5)
        random_area = np.random.uniform(0.05, 0.3)
        
       
        self.state = np.array(
            [random_x, random_y, random_area, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            dtype=np.float32
        )
        self.steps_left = 500
        
        return self.state, {}

    def step(self, action):
        roll, pitch, yaw, height_cmd = action
       
        current_x, current_y, current_area, l_f, l_b, l_l, l_r, l_u, l_d = self.state

        accel = np.random.uniform(-0.002, 0.002, size=2)
        self.person_velocity = (self.person_velocity + accel) * 0.98
        self.person_velocity = np.clip(self.person_velocity, -0.03, 0.03)
        
        person_dx = self.person_velocity[0]
        person_dy = self.person_velocity[1]
        
        self.person_world_pos[0] += person_dx
        self.person_world_pos[1] += person_dy
        self.person_world_pos[0] = np.clip(self.person_world_pos[0], -3, 3)
        self.person_world_pos[1] = np.clip(self.person_world_pos[1], -3, 3)

        correction_x = (roll * 0.05) 
        correction_y = (height_cmd * 0.05)
        correction_area = (pitch * 0.005)

        new_x = current_x - correction_x + person_dx
        new_y = current_y - correction_y
        new_area = current_area + correction_area - (person_dy * 0.1)
        new_area = max(0.05, min(0.8, new_area))
        
        dist_from_face = 1.0 / (new_area + 0.1) * 0.1
        drone_world_x = self.person_world_pos[0] - new_x
        drone_world_y = self.person_world_pos[1] - dist_from_face 
        drone_world_z = self.person_world_pos[2] + new_y

      
        ray_len = 2.0
        start_pos = [drone_world_x, drone_world_y, drone_world_z]
        
        directions = [
            [0, ray_len, 0],   # Front
            [0, -ray_len, 0],  # Back
            [-ray_len, 0, 0],  # Left
            [ray_len, 0, 0],   # Right
            [0, 0, ray_len],   # Up
            [0, 0, -ray_len]   # Down
        ]
        
        lidar_readings = []
        
        for d in directions:
            end_pos = [start_pos[0]+d[0], start_pos[1]+d[1], start_pos[2]+d[2]]
            results = p.rayTest(start_pos, end_pos)
            hit_fraction = results[0][2] 
            lidar_readings.append(hit_fraction)

       
        distance_penalty = -(new_x**2 + new_y**2)
        size_penalty = -((new_area - self.target_area)**2) * 10
       
        obstacle_penalty = 0
        min_dist = min(lidar_readings)
        if min_dist < 0.15: 
            obstacle_penalty = -5.0
        
        reward = distance_penalty + size_penalty + obstacle_penalty + 1.0
     
        self.state = np.array([new_x, new_y, new_area] + lidar_readings, dtype=np.float32)

      
        if self.render_mode == "human" and self.drone_id is not None:
            p.resetBasePositionAndOrientation(self.face_id, self.person_world_pos, [0,0,0,1])
            tilt = p.getQuaternionFromEuler([pitch*0.4, roll*0.4, 0])
            p.resetBasePositionAndOrientation(self.drone_id, [drone_world_x, drone_world_y, drone_world_z], tilt)
            
           

            time.sleep(1/60)

        self.steps_left -= 1
        terminated = False
        truncated = False
        
        if self.steps_left <= 0: truncated = True
        
       
        if abs(new_x) > 1.2 or abs(new_y) > 1.2 or min_dist < 0.05: 
            terminated = True
            reward -= 20
            
        return self.state, reward, terminated, truncated, {}
    
    def close(self):
        if self.physics_client: p.disconnect()
