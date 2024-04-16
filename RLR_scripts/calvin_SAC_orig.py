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
        return robot_obs[:7]

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
        robot_position = self.get_obs()
        return 5 - dist(robot_position[0:3] , [0.5, 0.5, 0.5]), r_info
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

def train(save_dir):
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

    log_dir = os.path.join(save_dir, "logs")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    average_returns = []
    return_indices = []
    average_steps = []
    for i in range(1000):
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name = "test", log_interval=4)
        episodes = 50
        total_score = 0.0
        total_steps = 0.0
        if i % 5 == 0:
            for episode in range(episodes):
                obs = env.reset()
                done = False
                score = 0.0
                steps = 0.0
                while not done and steps < 200:
                    obs, reward, done, info = env.step(env.action_space.sample())
                    score += reward/10.0
                    steps += 1
                total_score += score
                total_steps += steps
            average_score = total_score/episodes
            return_indices.append(i)
            average_returns.append(average_score)
            average_steps.append(total_steps/episodes)
            print(f"Episode: {i + 1}, Avg_Score: {average_score}, Avg_Steps: {total_steps/episodes}")
            # save image that plots average return vs episode
            
            # plot the average return vs indices
            plt.plot(return_indices, average_returns)
            plt.xlabel("Episode")
            plt.ylabel("Average Return")
            plt.title("Average Return vs Episode")
            plt.savefig(os.path.join(save_dir, "average_return_vs_episode.png"))
            plt.close()
        if i % 5 == 0:
            model_name = "sac_slide_time_" + str(i)
            model.save(os.path.join(checkpoint_dir, model_name))
    
    # save the average returns and average steps
    np.save(os.path.join(save_dir, "average_returns.npy"), average_returns)
    np.save(os.path.join(save_dir, "average_steps.npy"), average_steps)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a SAC model on the SlideEnv')
    parser.add_argument('--save_dir', 
                        type=str, 
                        required=True,
                        help='Directory to save the model')
    

    args = parser.parse_args()
    train(args.save_dir)