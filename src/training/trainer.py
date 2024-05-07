import numpy as np
import torch
import time


class Trainer:

    def __init__(self, plan_to_go, h_model, plan_encoder, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.plan_to_go = plan_to_go
        self.h_model = h_model
        self.plan_encoder = plan_encoder
        self.optimizer, self.plan_encoder_optimizer, self.l_optimizer = optimizer[0], optimizer[1], optimizer[2]
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler, self.plan_encoder_scheduler, self.l_scheduler = scheduler[0], scheduler[1], scheduler[2]
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()
        self.steps = []
        self.action_losses = []
        self.kl_losses = []
        self.returns = []
        self.lengths = []

    def train_iteration(self, results_dir, num_steps, iter_num=0, print_logs=False):

        logs = dict()

        train_start = time.time()

        self.h_model.train()
        self.plan_to_go.train()
        self.plan_encoder.train()
        for _ in range(num_steps):
            action_loss, kl_loss = self.train_step()
            self.action_losses.append(action_loss)
            self.kl_losses.append(kl_loss)
            self.steps.append(iter_num*num_steps+_)
            print("training steps:", str(iter_num*num_steps+_))
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.h_model.eval()
        self.plan_to_go.eval()
        self.plan_encoder.eval()

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(self.action_losses)
        logs['training/train_loss_std'] = np.std(self.action_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

