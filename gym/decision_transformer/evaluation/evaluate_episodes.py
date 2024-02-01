import numpy as np
import torch
import copy


def relabel_timestep(step, horizon):
    step_tmp = copy.deepcopy(step)
    add = 0
    step_tmp[0] = add + step[0] // horizon
    for i in range(1, len(step)):
        if step[i] < step[i - 1]:
            add = step_tmp[i - 1] + 1 if step[i] == 0 else step_tmp[i - 1]
        step_tmp[i] = add + step[i] // horizon
    return step_tmp


def evaluate_episode_rtg2(
        env,
        state_dim,
        act_dim,
        z_dim,
        h_model,
        plan_to_go,
        plan_encoder,
        horizon,
        K,
        episodes_times,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    context_size = K
    h_model.eval()
    h_model.to(device=device)
    plan_to_go.eval()
    plan_to_go.to(device=device)
    plan_encoder.eval()
    plan_encoder.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    ep_return = target_return
    sim_states = []
    episode_returns, episode_lengths = [],[]
    # h_timesteps = [0 for time_tmp in range(K)]
    h_timesteps = np.array([0])
    for episodes_time in range(episodes_times):
        print("eval_episode:  ",str(episodes_time))
        state = env.reset()
        state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        if episodes_time == 0:
            # states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            states = state.reshape(1, state_dim)
            # rewards = torch.zeros(0, device=device, dtype=torch.float32)
            z_distributions = torch.zeros((0, 2 * z_dim), device=device, dtype=torch.float32)
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        else:
            refresh_state = state.reshape(1, state_dim)
            refresh_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            states = torch.cat([states, refresh_state], dim=0)
            target_return = torch.cat([target_return, refresh_return], dim=1)
            # timesteps = torch.cat([timesteps,refresh_timesteps], dim=1)
            # del h_timesteps[0]
            h_timesteps = np.concatenate((h_timesteps,np.array([0])),axis=0)
        t, episode_return, episode_length = 0, 0, 0
        l_states = states[-1].reshape(1, -1)
        l_time_steps = timesteps[0, -1].reshape(-1, 1)
        l_actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        while t < max_ep_len:
            # add padding
            h_timesteps_tmp = relabel_timestep(copy.deepcopy(h_timesteps),horizon)
            h_timesteps_tmp = torch.from_numpy(np.array(h_timesteps_tmp).reshape(1,-1)).to(dtype=torch.long, device=device)
            z_distributions = torch.cat([z_distributions, torch.zeros((1, 2*z_dim), device=device)], dim=0)
            # rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            z_distribution_predict = h_model.get_action(
                # (states.to(dtype=torch.float32) - state_mean) / state_std,
                states.to(dtype=torch.float32),
                z_distributions.to(dtype=torch.float32),
                # rewards.to(dtype=torch.float32),
                None,
                target_return.to(dtype=torch.float32),
                # timesteps.to(dtype=torch.long),
                h_timesteps_tmp,
            )
            tmp_return = target_return[0,-1]
            z_distribution_predict_tmp = z_distribution_predict.unsqueeze(0).repeat([horizon,1])
            for i in range(horizon):
                l_actions = torch.cat([l_actions, torch.zeros((1, act_dim), device=device)], dim=0)
                action = plan_to_go.get_action(l_states, l_actions, z_distribution_predict_tmp,l_time_steps)
                l_actions[-1] = action.reshape(-1, act_dim)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
                t+=1

                cur_state = state.reshape(1, state_dim)
                if mode != 'delayed':
                    tmp_return = tmp_return - (reward/scale)
                else:
                    tmp_return = target_return[0,-1]
                episode_return += reward
                episode_length += 1
                if done:
                    normalized_score = env.get_normalized_score(episode_return)
                    episode_returns.append(normalized_score)
                    episode_lengths.append(episode_length)
                    break
                timesteps = torch.cat(
                    [timesteps,
                     torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
                # if i+1 < horizon:
                l_states = torch.cat([l_states, cur_state], dim=0)
                l_time_steps = torch.cat([l_time_steps, torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
            z_distributions[-1] = z_distribution_predict
            if done:
                # z_actual_distributions = plan_encoder.get_actual_distribution(l_states,l_actions,l_time_steps)
                # z_distributions[-1] = z_actual_distributions
                break
            # z_actual_distributions = plan_encoder.get_actual_distribution(l_states[:-1],l_actions,l_time_steps[:-1])
            # z_distributions[-1] = z_actual_distributions
            states = torch.cat([states, cur_state], dim=0)
            target_return = torch.cat(
                [target_return, tmp_return.reshape(1, 1)], dim=1)
            # del h_timesteps[0]
            # h_timesteps.append(t)
            h_timesteps = np.concatenate((h_timesteps, np.array([t])), axis=0)
    # episode_returns, episode_lengths = np.array(episode_returns), np.array(episode_lengths)
    return episode_returns, episode_lengths



def evaluate_episode_rtg3(
        env,
        state_dim,
        act_dim,
        z_dim,
        h_model,
        plan_to_go,
        plan_encoder,
        horizon,
        K,
        episodes_times,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    context_size = K
    h_model.eval()
    h_model.to(device=device)
    plan_to_go.eval()
    plan_to_go.to(device=device)
    plan_encoder.eval()
    plan_encoder.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    ep_return = target_return
    sim_states = []
    episode_returns, episode_lengths = [],[]
    # h_timesteps = [0 for time_tmp in range(K)]
    h_timesteps = np.array([0])
    for episodes_time in range(episodes_times):
        print("eval_episode:  ",str(episodes_time))
        state = env.reset()
        state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        if episodes_time == 0:
            # states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            states = state.reshape(1, state_dim)
            # rewards = torch.zeros(0, device=device, dtype=torch.float32)
            z_distributions = torch.zeros((0, 2 * z_dim), device=device, dtype=torch.float32)
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        else:
            refresh_state = state.reshape(1, state_dim)
            refresh_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            states = torch.cat([states, refresh_state], dim=0)
            target_return = torch.cat([target_return, refresh_return], dim=1)
            # timesteps = torch.cat([timesteps,refresh_timesteps], dim=1)
            # del h_timesteps[0]
            h_timesteps = np.concatenate((h_timesteps,np.array([0])),axis=0)
        t, episode_return, episode_length = 0, 0, 0
        l_states = states[-1].reshape(1, -1)
        l_time_steps = timesteps[0, -1].reshape(-1, 1)
        l_actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        while t < max_ep_len:
            # add padding

            h_timesteps_tmp = relabel_timestep(copy.deepcopy(h_timesteps),horizon)
            h_timesteps_tmp = torch.from_numpy(np.array(h_timesteps_tmp).reshape(1,-1)).to(dtype=torch.long, device=device)
            z_distributions = torch.cat([z_distributions, torch.zeros((1, 2*z_dim), device=device)], dim=0)
            # rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            z_distribution_predict = h_model.get_action(
                # (states.to(dtype=torch.float32) - state_mean) / state_std,
                states.to(dtype=torch.float32),
                z_distributions.to(dtype=torch.float32),
                # rewards.to(dtype=torch.float32),
                None,
                target_return.to(dtype=torch.float32),
                # timesteps.to(dtype=torch.long),
                h_timesteps_tmp,
            )
            tmp_return = target_return[0,-1]
            z_distribution_predict_tmp = z_distribution_predict.unsqueeze(0).repeat([horizon,1])
            for i in range(horizon):
                l_actions = torch.cat([l_actions, torch.zeros((1, act_dim), device=device)], dim=0)
                action = plan_to_go.get_action(l_states, l_actions, z_distribution_predict_tmp,l_time_steps)
                l_actions[-1] = action.reshape(-1, act_dim)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
                t+=1
                # cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                cur_state = state.reshape(1, state_dim)
                # rewards[-1] = reward
                if mode != 'delayed':
                    tmp_return = tmp_return - (reward/scale)
                else:
                    tmp_return = target_return[0,-1]
                episode_return += reward
                episode_length += 1
                if done:
                    normalized_score = env.get_normalized_score(episode_return)
                    episode_returns.append(normalized_score)
                    episode_lengths.append(episode_length)
                    break
                timesteps = torch.cat(
                    [timesteps,
                     torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
                # if i+1 < horizon:
                l_states = torch.cat([l_states, cur_state], dim=0)
                l_time_steps = torch.cat([l_time_steps, torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)
            # z_distributions[-1] = z_distribution_predict
            if done:
                z_actual_distributions = plan_encoder.get_actual_distribution(l_states,l_actions,l_time_steps)
                z_distributions[-1] = z_actual_distributions
                break
            z_actual_distributions = plan_encoder.get_actual_distribution(l_states[:-1],l_actions,l_time_steps[:-1])
            z_distributions[-1] = z_actual_distributions
            states = torch.cat([states, cur_state], dim=0)
            target_return = torch.cat(
                [target_return, tmp_return.reshape(1, 1)], dim=1)
            # del h_timesteps[0]
            # h_timesteps.append(t)
            h_timesteps = np.concatenate((h_timesteps, np.array([t])), axis=0)
    # episode_returns, episode_lengths = np.array(episode_returns), np.array(episode_lengths)
    return episode_returns, episode_lengths
