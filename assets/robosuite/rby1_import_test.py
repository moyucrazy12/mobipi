import numpy as np
import robocasa
import robosuite.models.robots.manipulators.rby1_robot
from robosuite.controllers import load_composite_controller_config
import math

controller_config = load_composite_controller_config(robot="RBY1")

env = robocasa.make(
    env_name="Lift", # CloseDrawer (RoboCasa environment), ...
    robots="RBY1",
    controller_configs=controller_config,
    initialization_noise=None,
    use_camera_obs=False,
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera='frontview', # robot0_head_camera, robot0_right_eye_in_hand, robot0_left_eye_in_hand
)

env.reset()

# 1. Setup parameters for the waving motion
step_counter = 0
frequency = 0.01  # Controls how fast the robot waves
amplitude = 1.0   # Controls how wide the wave is

# 2. Target the specific joint (Index 1 or 2 is usually the shoulder roll)
wave_joint_idx = 1 

for _ in range(1000):
    # Start with a baseline of zero movement for all joints
    action = np.zeros(env.action_dim)
    
    # Calculate the smooth back-and-forth sine wave
    wave_signal = math.sin(step_counter * frequency) * amplitude
    
    # Apply the sine wave ONLY to the targeted joint
    action[wave_joint_idx] = wave_signal
    
    _, reward, done, _ = env.step(action)
    env.render()
    
    step_counter += 1

    if done:
        env.reset()
        step_counter = 0

env.close()