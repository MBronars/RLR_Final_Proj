from calvin_env.envs.play_table_env import PlayTableSimEnv
from hydra import initialize, compose
import time
import hydra
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from gym import spaces
import gym
import numpy as np
from stable_baselines3 import SAC

import imageio

class SlideEnv(PlayTableSimEnv):
    def __init__(self,
                 tasks: dict = {},
                 **kwargs):
        super(SlideEnv, self).__init__(**kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,))
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        return obs

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        robot_obs, robot_info = self.robot.get_observation()
        scene_obs = self.scene.get_obs()
        slider_obs = scene_obs[0]
        # append slider observation to robot observation np array
        return np.append(robot_obs[:7], slider_obs)

    def _success(self):
        """ Returns a boolean indicating if the task was performed correctly """
        current_info = self.get_info()
        task_filter = ["move_slider_left"]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return 'move_slider_left' in task_info

    def _reward(self):
        """ Returns the reward function that will be used 
        for the RL algorithm """
        reward = int(self._success()) * 10
        r_info = {'reward': reward}
        return reward, r_info

    def _termination(self):
        """ Indicates if the robot has reached a terminal state """
        success = self._success()
        done = success
        d_info = {'success': success}        
        return done, d_info

    def step(self, action):
            """ Performing a relative action in the environment
                input:
                    action: 7 tuple containing
                            Position x, y, z. 
                            Angle in rad x, y, z. 
                            Gripper action
                            each value in range (-1, 1)

                            OR
                            8 tuple containing
                            Relative Joint angles j1 - j7 (in rad)
                            Gripper action
                output:
                    observation, reward, done info
            """
            # Transform gripper action to discrete space
            env_action = action.copy()
            env_action[-1] = (int(action[-1] >= 0) * 2) - 1

            # for using actions in joint space
            if len(env_action) == 8:
                env_action = {"action": env_action, "type": "joint_rel"}

            self.robot.apply_action(env_action)
            for i in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
            obs = self.get_obs()
            info = self.get_info()
            reward, r_info = self._reward()
            done, d_info = self._termination()
            info.update(r_info)
            info.update(d_info)
            return obs, reward, done, info
def capture_frame(env):
    # Example camera position and orientation
    camera_pos = [1, 0, 1]  #adjustment 
    target_pos = [0, 0, 0]  #adjustment
    up_vector = [0, 0, 1]  # Typically the z-axis is up

    view_matrix = env.p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vector
    )

    projection_matrix = env.p.computeProjectionMatrixFOV(
        fov=60, aspect=float(640)/480,
        nearVal=0.1, farVal=100.0
    )

    _, _, img, _, _ = env.p.getCameraImage(width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix, renderer=env.p.ER_BULLET_HARDWARE_OPENGL)
    return img[:, :, :3]


def rollout():
    with initialize(config_path="../../../calvin/calvin_env/conf/"):
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        cfg.env["use_egl"] = False
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True
        print(cfg.env)

    new_env_cfg = {**cfg.env}
    new_env_cfg["tasks"] = cfg.tasks
    new_env_cfg.pop('_target_', None)
    new_env_cfg.pop('_recursive_', None)
    env = SlideEnv(**new_env_cfg)
    
    log_dir = "/srv/rl2-lab/flash8/mbronars3/workspace/RLR/SAC_Task_Buff/temp_logs"

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.load("/srv/rl2-lab/flash8/mbronars3/workspace/RLR/SAC_DenseReward/checkpoints/sac_slide_time_5.zip")

    episodes = 1
    frames = []
    for episode in range(episodes):
        #video creation
        obs = env.reset()
        done = False
        counter = 0
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            print(obs[-1])
            frame= capture_frame(env)
            frames.append(frame)
            if counter >= 300:
                break
            counter += 1
    
    video_path = "/srv/rl2-lab/flash8/mbronars3/workspace/RLR/SAC_Task_Buff/videos/test.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved at {video_path}")


if __name__ == "__main__":
    # Parse arguments
    # parser = argparse.ArgumentParser(description='Train a SAC model on the SlideEnv')
    # parser.add_argument('--save_dir', 
    #                     type=str, 
    #                     required=True,
    #                     help='Directory to save the model')
    

    # args = parser.parse_args()
    # train(args.save_dir)
    rollout()