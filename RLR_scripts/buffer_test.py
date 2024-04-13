import stable_baselines3
from stable_baselines3.common.buffers import ReplayBuffer
import os
import argparse
import numpy as np
import gymnasium

# Write main function for testing out the buffer, take locatin of dataset as command line argument
def test_buffer(dataset_path, buffer_size=100):
    # Get all .npz files in the dataset
    npz_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]

    # Format of files is episode_x.npz, order the list by episode number
    npz_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # load in .npy file "ep_start_end_ids.npy"
    ep_start_end_ids = np.load(os.path.join(dataset_path, "ep_start_end_ids.npy"))

    avoid_indices = []

    # get index of all episodes that are end episodes
    for i in range(len(ep_start_end_ids)):
        end_ep = ep_start_end_ids[i][1]
        # append 0's to beginning until ep_end is 7 digits
        end_ep = f"{end_ep:07d}"
        # get index of end episode
        file_name = f"episode_{end_ep}.npz"
        avoid_indices.append(npz_files.index(file_name))

    total_episodes = np.arange(len(npz_files))
    good_indices = np.setdiff1d(total_episodes, avoid_indices)

    # Randomly sample from good_indices
    random_indices = np.random.choice(good_indices, buffer_size, replace=False)

    test_data = np.load(os.path.join(dataset_path, npz_files[0]))
    obs_shape = (len(test_data['robot_obs']) + len(test_data['scene_obs']), )
    action_shape = (len(test_data['actions']),)
    obs_space = gymnasium.spaces.Box(-1, 1, shape = obs_shape, dtype = np.float32)
    action_space = gymnasium.spaces.Box(-1, 1, shape = action_shape, dtype = np.float32)


    test_buff = ReplayBuffer(buffer_size, observation_space = obs_space, action_space = action_space)


    for i in random_indices:
        # get episode number
        file = npz_files[i]
        episode = int(file.split('_')[1].split('.')[0])

        data = np.load(os.path.join(dataset_path, file))
        next_data = np.load(os.path.join(dataset_path, npz_files[i+1]))

        action = data['actions']
        robot_obs = data['robot_obs']
        scene_obs = data['scene_obs']
        next_robot_obs = next_data['robot_obs']
        next_scene_obs = next_data['scene_obs']

        obs = np.concatenate((robot_obs, scene_obs), axis=-1)
        next_obs = np.concatenate((next_robot_obs, next_scene_obs), axis=-1)

        reward = np.array([1])
        done = np.array([0])
        infos = [{}]

        # print shapes of all data
        print("Action shape: ", action.shape)
        print("Obs shape: ", obs.shape)
        print("Next obs shape: ", next_obs.shape)
        print("Reward shape: ", reward.shape)
        print("Done shape: ", done.shape)

        test_buff.add(obs = obs, action = action, reward = reward, next_obs = next_obs, done = done, infos = infos)

    from IPython import embed; embed()

    
    

    

    

    # Test out buffer

# Read in command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Location of dataset training folder")
args = parser.parse_args()
dataset = args.dataset

test_buffer(dataset)