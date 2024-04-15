import os
import argparse
import numpy as np
import gymnasium
from stable_baselines3.common.buffers import ReplayBuffer

def test_buffer(dataset_path, buffer_size=100):
    npz_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]
    npz_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    test_buff = None
    last_slider_position = 1

    ep_start_end_ids = np.load(os.path.join(dataset_path, "ep_start_end_ids.npy"))
    start_indices = ep_start_end_ids[:, 0]
    end_indices = ep_start_end_ids[:, 1]

    for i, file_name in enumerate(npz_files):
        if i in end_indices:
            continue
        data = np.load(os.path.join(dataset_path, file_name))

        scene_obs = data['scene_obs']
        robot_obs = data['robot_obs']
        actions = data['actions']

        current_slider_position = scene_obs[0]  # Assuming slider position is the first element in scene_obs

        # Check if the slider was closed in the last observation and is now open
        if last_slider_position <= 0.25 and current_slider_position > 0.25:
            # This is the condition that a relevant movement has occurred
            # Calculate the range of indices to include in the buffer
            start = i - 30  # Ensure start is not negative
            end = i + 1  # Include current observation
            

            if test_buff is None:
                # Initialize buffer with correct shapes
                obs_shape = (robot_obs.shape[1][:7],)
                action_shape = (actions.shape[1],)
                obs_space = gymnasium.spaces.Box(-1, 1, shape=obs_shape, dtype=np.float32)
                action_space = gymnasium.spaces.Box(-1, 1, shape=action_shape, dtype=np.float32)
                test_buff = ReplayBuffer(buffer_size, observation_space=obs_space, action_space=action_space)

            #check if there is a start_index or end_index in the range of indices, if so change the start index to that index
            for j in range(start, end):
                if j in start_indices:
                    start = j
                if j in end_indices:
                    start = j + 1

            for j in range(start, end):
                file = npz_files[j]
                data = np.load(os.path.join(dataset_path, file))
                next_data = np.load(os.path.join(dataset_path, npz_files[j + 1]))

                actions = data['actions']
                robot_obs = data['robot_obs']
                scene_obs = data['scene_obs']
                next_robot_obs = next_data['robot_obs']
                next_scene_obs = next_data['scene_obs']

                obs = robot_obs[:7]
                next_obs = next_robot_obs[:7]
                done = np.array([False])
                infos = [{'TimeLimit.truncated': False}]
                test_buff.add(obs=obs[0], action=actions[0], reward=np.ones(1), next_obs=next_obs[0], done=done, infos=infos)

        last_slider_position = current_slider_position

    if test_buff is not None and test_buff.size() > 0:
        print("Buffer size:", test_buff.size())
        print("Sample observation:", test_buff.observations[0])
    else:
        print("No relevant movement detected in any episodes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Location of dataset training folder")
    args = parser.parse_args()
    dataset = args.dataset

    test_buffer(dataset)