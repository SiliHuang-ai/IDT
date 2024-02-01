import gym
import numpy as np

import collections
import pickle

import d4rl


datasets = []

# for env_name in ['halfcheetah', 'hopper', 'walker2d']:
for env_name in ['walker2d']:
	# for dataset_type in ['medium', 'medium-replay', 'expert', 'medium-expert']:
	for dataset_type in ['medium-expert']:
		name = f'{env_name}-{dataset_type}-v2'
		env = gym.make(name)
		dataset = env.get_dataset()

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset:
			use_timeouts = True

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == 1000-1)
			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
				data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(f'{name}.pkl', 'wb') as f:
			pickle.dump(paths, f)

'''
Warning: Flow failed to import. Set the environment variable D4RL_SUPPRESS_IMPORT_ERROR=1 to suppress this message.
No module named 'flow'
/home/hujifeng/anaconda3/envs/DT_env/lib/python3.8/site-packages/glfw/__init__.py:912: GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
  warnings.warn(message, GLFWError)
Warning: CARLA failed to import. Set the environment variable D4RL_SUPPRESS_IMPORT_ERROR=1 to suppress this message.
No module named 'carla'
pybullet build time: May 20 2022 19:44:17
/home/hujifeng/anaconda3/envs/DT_env/lib/python3.8/site-packages/gym/logger.py:30: UserWarning: WARN: Box bound precision lowered by casting to float32
  warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5 to /home/hujifeng/.d4rl/datasets/halfcheetah_medium-v2.hdf5
load datafile: 100%|████████████████████████████| 21/21 [00:01<00:00, 12.75it/s]
Number of samples collected: 1000000
Trajectory returns: mean = 4770.3349609375, std = 355.7503967285156, max = 5309.37939453125, min = -310.23419189453125
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5 to /home/hujifeng/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5
load datafile: 100%|████████████████████████████| 11/11 [00:00<00:00, 31.84it/s]
Number of samples collected: 202000
Trajectory returns: mean = 3093.28564453125, std = 1680.6939697265625, max = 4985.1416015625, min = -638.4852905273438
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5 to /home/hujifeng/.d4rl/datasets/halfcheetah_expert-v2.hdf5
load datafile: 100%|████████████████████████████| 21/21 [00:01<00:00, 12.88it/s]
Number of samples collected: 1000000
Trajectory returns: mean = 10656.42578125, std = 441.6827087402344, max = 11252.03515625, min = 2045.8277587890625
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5 to /home/hujifeng/.d4rl/datasets/hopper_medium-v2.hdf5
load datafile: 100%|████████████████████████████| 21/21 [00:01<00:00, 19.30it/s]
Number of samples collected: 999906
Trajectory returns: mean = 1422.05615234375, std = 378.9537048339844, max = 3222.360595703125, min = 315.8680114746094
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5 to /home/hujifeng/.d4rl/datasets/hopper_medium_replay-v2.hdf5
load datafile: 100%|████████████████████████████| 11/11 [00:00<00:00, 24.80it/s]
Number of samples collected: 402000
Trajectory returns: mean = 467.3020324707031, std = 511.0256042480469, max = 3192.925048828125, min = -1.4400691986083984
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5 to /home/hujifeng/.d4rl/datasets/hopper_expert-v2.hdf5
load datafile: 100%|████████████████████████████| 21/21 [00:01<00:00, 19.39it/s]
Number of samples collected: 999494
Trajectory returns: mean = 3511.357666015625, std = 328.5859680175781, max = 3759.083740234375, min = 1645.2764892578125
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5 to /home/hujifeng/.d4rl/datasets/walker2d_medium-v2.hdf5
load datafile: 100%|████████████████████████████| 21/21 [00:01<00:00, 12.65it/s]
Number of samples collected: 999995
Trajectory returns: mean = 2852.08837890625, std = 1095.443359375, max = 4226.93994140625, min = -6.6056718826293945
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5 to /home/hujifeng/.d4rl/datasets/walker2d_medium_replay-v2.hdf5
load datafile: 100%|████████████████████████████| 11/11 [00:00<00:00, 21.39it/s]
Number of samples collected: 302000
Trajectory returns: mean = 682.7012939453125, std = 895.95556640625, max = 4132.00048828125, min = -50.196834564208984
Downloading dataset: http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5 to /home/hujifeng/.d4rl/datasets/walker2d_expert-v2.hdf5
load datafile: 100%|████████████████████████████| 21/21 [00:01<00:00, 12.37it/s]
Number of samples collected: 999214
Trajectory returns: mean = 4920.5068359375, std = 136.3949432373047, max = 5011.693359375, min = 763.4161376953125

Process finished with exit code 0
'''
