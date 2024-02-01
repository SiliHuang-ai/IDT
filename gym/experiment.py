import gym
import d4rl
import numpy as np
import torch
import wandb
import os

import argparse
import pickle
import random
import sys
import itertools

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg2, evaluate_episode_rtg3
from decision_transformer.models.decision_transformer import HDecisionTransformer
from decision_transformer.models.primitive_policy import PrimitivePolicy
from decision_transformer.models.decoder_action import DecoderAction
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def load(h_model, l_model, model_dir, step):
    h_model.load_state_dict(
        torch.load('%s/h_policy_%s.pt' % (model_dir, step))
    )
    l_model.load_state_dict(
        torch.load('%s/l_policy_%s.pt' % (model_dir, step))
    )
    return h_model, l_model

def save(h_model, l_model, model_dir, step):
    torch.save(
        h_model.state_dict(), '%s/h_policy_%s.pt' % (model_dir, step)
    )
    torch.save(
        l_model.state_dict(), '%s/l_policy_%s.pt' % (model_dir, step)
    )

def experiment(
        exp_prefix,
        variant,
):

    device = variant['device']
    # log_to_wandb = variant.get('log_to_wandb', True)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    # group_name = f'{exp_prefix}-{env_name}-{dataset}'
    # exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    make_dir(variant['work_dir']+f'{env_name}-{dataset}')
    results_dir = make_dir(os.path.join(args.work_dir,'results'))
    model_dir = make_dir(os.path.join(variant['work_dir'], 'models'))

    if env_name == 'hopper':
        # env = gym.make('Hopper-v3')
        env = gym.make(f'{env_name}-{dataset}-v2')
        # env = gym.make('hopper-medium-v0')
        max_ep_len = 1000
        # env_targets = [3600, 1800]  # evaluation conditioning targets
        env_targets = [3600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        # env = gym.make('HalfCheetah-v3')
        env = gym.make(f'{env_name}-{dataset}-v2')
        max_ep_len = 1000
        # env_targets = [12000, 6000]
        env_targets = [12000]
        scale = 1000.
    elif env_name == 'walker2d':
        # env = gym.make('Walker2d-v3')
        env =env = gym.make(f'{env_name}-{dataset}-v2')
        max_ep_len = 1000
        # env_targets = [5000, 2500]
        env_targets = [5000]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    z_dim = variant['z_dim']

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)


    K = variant['K']
    horizon = variant['horizon']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    # sorted_inds = sorted_inds[-num_trajectories:]
    top_sorted_inds = sorted_inds[-num_trajectories:]
    low_sorted_inds = sorted_inds[:len(trajectories)-num_trajectories]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    # p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    top_p_sample = traj_lens[top_sorted_inds] / sum(traj_lens[top_sorted_inds])
    low_p_sample = traj_lens[low_sorted_inds] / sum(traj_lens[low_sorted_inds])
    min_top_length = np.min(traj_lens[top_sorted_inds])//horizon
    mean_top_return = np.mean(returns[top_sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        top_batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=top_p_sample,  # reweights so we sample according to timesteps
        )
        top_batch_inds = top_batch_inds[np.argsort(top_batch_inds)]

        low_batch_inds = np.random.choice(
            np.arange(len(trajectories)-num_trajectories),
            size=batch_size,
            replace=True,
            p=low_p_sample,  # reweights so we sample according to timesteps
        )
        low_batch_inds = low_batch_inds[np.argsort(low_batch_inds)]

        h_s, h_rtg, h_difference, h_timesteps, h_mask, h_loss_mask, s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], [], [], [], [], [], [], []
        # for tmp in range(batch_size):
        min_rest_batch,max_rest_batch = np.arange(batch_size),np.arange(batch_size)
        i=None
        # traj = trajectories[int(sorted_inds[batch_inds[i]])]
        while len(min_rest_batch)>0: #每次就用batch_size个轨迹
            h_s_tmp, h_rtg_tmp, h_timesteps_tmp = [],[],[]
            idx=0
            while len(h_s_tmp)<(max_len-min_top_length) and idx<len(min_rest_batch):
            # get high-level sequences from dataset
                i = min_rest_batch[idx]
                traj = trajectories[int(low_sorted_inds[low_batch_inds[i]])]
                # si = random.randint(0, traj['rewards'].shape[0] - 1)
                for si in range(0,len(traj['observations']),horizon):
                    h_s_tmp.append(traj['observations'][si:si+1].reshape(1, -1, state_dim))
                    h_timesteps_tmp.append(np.arange(si, si+1).reshape(1,-1))
                    tmp = discount_cumsum(traj['rewards'][si:], gamma=1.)
                    # [:s[-1].shape[1] + 1].reshape(1, -1, 1)
                    h_rtg_tmp.append(tmp[0].reshape(1,-1,1))
                    # h_rtg_tmp.append(tmp[:,0,:].reshape(1,-1,1))

                    # get low-level sequences following si at the end of high-level
                    s.append(traj['observations'][si:si + horizon].reshape(1, -1, state_dim))
                    a.append(traj['actions'][si:si + horizon].reshape(1, -1, act_dim))
                    r.append(traj['rewards'][si:si + horizon].reshape(1, -1, 1))
                    if 'terminals' in traj:
                        d.append(traj['terminals'][si:si + horizon].reshape(1, -1))
                    else:
                        d.append(traj['dones'][si:si + horizon].reshape(1, -1))
                    timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                    timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
                    rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                    if rtg[-1].shape[1] <= s[-1].shape[1]:
                        rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                    # padding and state + reward normalization
                    tlen = s[-1].shape[1]

                    s[-1] = np.concatenate([np.zeros((1, horizon - tlen, state_dim)), s[-1]], axis=1)
                    s[-1] = (s[-1] - state_mean) / state_std
                    a[-1] = np.concatenate([np.ones((1, horizon - tlen, act_dim)) * -10., a[-1]], axis=1)
                    r[-1] = np.concatenate([np.zeros((1, horizon - tlen, 1)), r[-1]], axis=1)
                    d[-1] = np.concatenate([np.ones((1, horizon - tlen)) * 2, d[-1]], axis=1)
                    rtg[-1] = np.concatenate([np.zeros((1, horizon - tlen, 1)), rtg[-1]], axis=1) / scale
                    timesteps[-1] = np.concatenate([np.zeros((1, horizon - tlen)), timesteps[-1]], axis=1)
                    mask.append(np.concatenate([np.zeros((1, horizon - tlen)), np.ones((1, tlen))],
                                               axis=1))
                    # if len(h_s_tmp)==max_len:
                    #     break
                start = idx
                min_rest_batch = np.delete(min_rest_batch, idx)
                if start<len(min_rest_batch):
                    idx = np.random.randint(start, len(min_rest_batch))
                else: idx = len(min_rest_batch)

            max_idx = np.random.randint(0, len(max_rest_batch))
            max_i = max_rest_batch[max_idx]
            max_rest_batch = np.delete(max_rest_batch,max_idx)
            traj = trajectories[int(top_sorted_inds[top_batch_inds[max_i]])]
            for si in range(0, len(traj['observations']), horizon):
                h_s_tmp.append(traj['observations'][si:si + 1].reshape(1, -1, state_dim))
                h_timesteps_tmp.append(np.arange(si, si + 1).reshape(1, -1))
                tmp = discount_cumsum(traj['rewards'][si:], gamma=1.)
                # [:s[-1].shape[1] + 1].reshape(1, -1, 1)
                h_rtg_tmp.append(tmp[0].reshape(1, -1, 1))
                # h_rtg_tmp.append(tmp[:,0,:].reshape(1,-1,1))

                # get low-level sequences following si at the end of high-level
                s.append(traj['observations'][si:si + horizon].reshape(1, -1, state_dim))
                a.append(traj['actions'][si:si + horizon].reshape(1, -1, act_dim))
                r.append(traj['rewards'][si:si + horizon].reshape(1, -1, 1))
                if 'terminals' in traj:
                    d.append(traj['terminals'][si:si + horizon].reshape(1, -1))
                else:
                    d.append(traj['dones'][si:si + horizon].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
                rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate([np.zeros((1, horizon - tlen, state_dim)), s[-1]], axis=1)
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate([np.ones((1, horizon - tlen, act_dim)) * -10., a[-1]], axis=1)
                r[-1] = np.concatenate([np.zeros((1, horizon - tlen, 1)), r[-1]], axis=1)
                d[-1] = np.concatenate([np.ones((1, horizon - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = np.concatenate([np.zeros((1, horizon - tlen, 1)), rtg[-1]], axis=1) / scale
                timesteps[-1] = np.concatenate([np.zeros((1, horizon - tlen)), timesteps[-1]], axis=1)
                mask.append(np.concatenate([np.zeros((1, horizon - tlen)), np.ones((1, tlen))],
                                           axis=1))


            h_rtg_tmp, h_loss_mask_tmp, max_difference = relabel_rtg(h_rtg_tmp[:],h_timesteps_tmp,max_len)
            # _, h_loss_mask_tmp, max_difference = relabel_rtg(h_rtg_tmp[:],h_timesteps_tmp,max_len)
            needed_delete = len(h_s_tmp) - max_len
            if needed_delete > 0:
                low_level_insert = len(s) - len(h_s_tmp)
                for tmp in range(needed_delete):
                    del h_s_tmp[0]
                    del h_rtg_tmp[0]
                    del h_timesteps_tmp[0]
                    del s[low_level_insert]
                    del a[low_level_insert]
                    del r[low_level_insert]
                    del d[low_level_insert]
                    del rtg[low_level_insert]
                    del timesteps[low_level_insert]
                    del mask[low_level_insert]

            h_s.append(np.concatenate(h_s_tmp, axis=1))
            h_timesteps_tmp = relabel_timestep(h_timesteps_tmp,horizon)
            h_difference.append(max_difference)
            h_loss_mask.append(h_loss_mask_tmp)
            h_rtg.append(np.concatenate(h_rtg_tmp, axis=1))
            h_timesteps.append(np.concatenate(h_timesteps_tmp, axis=1))
            # high-level 的 padding and state + reward normalization
            tlen = h_s[-1].shape[1]
            needed_padding = max_len-tlen
            if needed_padding>0:
                insert_idx = len(s)-tlen
                for j in range(needed_padding):
                    s.insert(insert_idx,np.zeros((1, horizon, state_dim)))
                    s[insert_idx] = (s[insert_idx] - state_mean) / state_std
                    a.insert(insert_idx,np.ones((1, horizon, act_dim)) * -10.)
                    d.insert(insert_idx,np.ones((1, horizon)))
                    r.insert(insert_idx,np.zeros((1, horizon, 1)))
                    rtg.insert(insert_idx,np.zeros((1, horizon+1, 1)) / scale)
                    timesteps.insert(insert_idx,np.zeros((1, horizon)))
                    mask.insert(insert_idx,np.zeros((1, horizon)))
            h_s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), h_s[-1]], axis=1)
            h_s[-1] = (h_s[-1] - state_mean) / state_std
            h_rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), h_rtg[-1]], axis=1) / scale
            # h_loss_mask[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), h_rtg[-1]], axis=1)
            h_timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), h_timesteps[-1]], axis=1)
            h_mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))],
                                           axis=1))


        h_difference = torch.from_numpy(np.concatenate(h_difference, axis=0)).to(dtype=torch.float32, device=device)
        h_s =torch.from_numpy(np.concatenate(h_s, axis=0)).to(dtype=torch.float32, device=device)
        h_rtg = torch.from_numpy(np.concatenate(h_rtg, axis=0)).to(dtype=torch.float32, device=device)
        h_timesteps = torch.from_numpy(np.concatenate(h_timesteps, axis=0)).to(dtype=torch.long, device=device)
        h_mask = torch.from_numpy(np.concatenate(h_mask, axis=0)).to(dtype=torch.float32,device=device)
        h_loss_mask = torch.from_numpy(np.concatenate(h_loss_mask, axis=0)).to(dtype=torch.float32,device=device)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.float32,device=device)

        return s, a, r, d, rtg, timesteps, mask, h_s, h_rtg, h_timesteps, h_mask, h_loss_mask, h_difference

    def relabel_rtg(rtg,step,max_len):
        max_rtg_idx = len(rtg)-1
        while step[max_rtg_idx]>step[max_rtg_idx-1]:
            max_rtg_idx -= 1
        h_loss_mask_tmp = np.concatenate([np.zeros((1, max_len-len(step)+max_rtg_idx)), np.ones((1, len(step)-max_rtg_idx))],axis=1)
        max_rtg = rtg[max_rtg_idx]
        max_difference = max_rtg-rtg[0]
        tmp = max_rtg_idx-1
        for idx in range(max_rtg_idx-1,-1,-1):
            if step[idx]<step[idx-1]:
                add = max_rtg - rtg[idx]
                for add_idx in range(tmp,idx-1,-1):
                    rtg[add_idx] += add
                tmp = idx-1
        return rtg, h_loss_mask_tmp, max_difference

    def relabel_timestep(step,horizon):
        step_tmp = step[:]
        add = 0
        step_tmp[0] = add + step[0]//horizon
        for i in range(1,len(step)):
            if step[i]<step[i-1]:
                add = step_tmp[i-1]+1 if step[i]==0 else step_tmp[i-1]
            step_tmp[i] = add + step[i]//horizon
        return step_tmp
    def eval_episodes(target_rew):
        def fn(h_model,plan_to_go,plan_encoder,rtg):
            episodes_times = 10
            eval_dict = dict()
            returns_evl, lengths = [[] for i in range(episodes_times)], [[] for i in range(episodes_times)]
            if rtg==1:
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        ret, length = evaluate_episode_rtg2(
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
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    print('rtg: '+str(rtg)+'    eval_times: ',str(_))
                    for i in range(episodes_times):
                        returns_evl[i].append(ret[i])
                        lengths[i].append(length[i])
            else:
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        ret, length = evaluate_episode_rtg3(
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
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    print('rtg: ' + str(rtg) + '    eval_times: ', str(_))
                    for i in range(episodes_times):
                        returns_evl[i].append(ret[i])
                        lengths[i].append(length[i])

            return returns_evl, lengths, target_rew

        return fn

    h_model = HDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        z_dim=variant['z_dim'],
        max_length=K,
        # max_ep_len=10,
        max_ep_len=K+max_ep_len//horizon,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )

    h_model = h_model.to(device=device)

    Plan_to_go = DecoderAction(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=variant['embed_dim'],
        z_dim=variant['z_dim'],
        max_length=horizon,
        max_ep_len=max_ep_len,
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    Plan_to_go = Plan_to_go.to(device=device)
    # target_action_decoder = DecoderAction(
    #     state_dim=state_dim,
    #     act_dim=act_dim,
    #     z_dim = variant['z_dim'],
    #     hidden_size=variant['embed_dim']
    # )
    # target_action_decoder = target_action_decoder.to(device=device)
    Plan_encoder = PrimitivePolicy(
        # decoder_action=action_decoder,
        # target_decoder_action=target_action_decoder,
        state_dim=state_dim,
        act_dim=act_dim,
        z_dim = variant['z_dim'],
        max_length=horizon,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    Plan_encoder = Plan_encoder.to(device=device)
    # decoder_actions_copy =

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        itertools.chain(h_model.parameters(),Plan_to_go.parameters(),Plan_encoder.parameters()),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    # h_optimizer = torch.optim.AdamW(
    #     h_model.parameters(),
    #     lr=variant['learning_rate'],
    #     weight_decay=variant['weight_decay'],
    # )
    # h_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     h_optimizer,
    #     lambda steps: min((steps+1)/warmup_steps, 1)
    # )
    # Plan_encoder_optimizer = torch.optim.AdamW(
    #     Plan_encoder.parameters(),
    #     lr=variant['learning_rate'],
    #     weight_decay=variant['weight_decay'],
    # )
    # Plan_encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     Plan_encoder_optimizer,
    #     lambda steps: min((steps + 1) / warmup_steps, 1)
    # )

    trainer = SequenceTrainer(
        plan_to_go=Plan_to_go,
        # target_decoder_action=target_action_decoder,
        h_model=h_model,
        plan_encoder=Plan_encoder,
        # optimizer=[optimizer,h_optimizer,l_optimizer],
        optimizer=[optimizer,None,None],
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=[scheduler,None,None],
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    # if log_to_wandb:
    #     try:
    #         wandb.init(
                # name=exp_prefix,
                # group=group_name,
        #         project='hsl_hdt_0',
        #         config=variant
        #     )
        # except: exit()
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        # if log_to_wandb:
        #     wandb.init(
        #         name=exp_prefix,
        #         group=group_name,
        #         project='hsl_hdt',
        #         config=variant
        #     )
        outputs = trainer.train_iteration(results_dir, num_steps=variant['num_steps_per_iter'], iter_num=iter, print_logs=True)
        save(h_model,Plan_to_go,model_dir=model_dir,step=0)
        # if log_to_wandb:
        #     wandb.log(outputs)
        # wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah')#hopper,halfcheetah,walker2d
    parser.add_argument('--dataset', type=str, default='medium-expert')  # medium, medium-replay, medium-expert, expert, random
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--horizon', type=int, default=10)#
    parser.add_argument('--K', type=int, default=300)#50
    parser.add_argument('--pct_traj', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)#64
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1)#
    parser.add_argument('--num_eval_episodes', type=int, default=10)#100
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)#10000
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--work_dir', default='.', type=str)
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
