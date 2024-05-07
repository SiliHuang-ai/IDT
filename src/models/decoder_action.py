import numpy as np
import torch
import torch.nn as nn
import transformers

from src.models.model import TrajectoryModel
from src.models.trajectory_gpt2 import GPT2Model
from torch.autograd import Variable


class DecoderAction(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            z_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.z_dim=z_dim
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_z = torch.nn.Linear(z_dim, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )


    def forward(self, states, actions, z, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        z_embeddings = self.embed_z(z)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        z_embeddings = z_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (z_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:,1])

        return action_preds

    def get_predict_action(self, states, actions, z_distributions, timesteps, attention_mask=None):

        z_distributions = z_distributions.reshape(-1, 1, 2 * self.z_dim).repeat(1, states.size()[1], 1)

        mu = z_distributions[:, :, :self.z_dim]
        logvar = z_distributions[:, :, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        action_preds = self.forward(states, actions, z, timesteps, attention_mask)

        return action_preds

    def get_action(self,states, actions, z_distributions, timesteps):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        z_distributions = z_distributions.reshape(1, -1, 2*self.z_dim)
        mu = z_distributions[:, :, :self.z_dim]
        logvar = z_distributions[:, :, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            z = z[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)

            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        action_preds = self.forward(states, actions, z, timesteps, attention_mask)
        return action_preds[0,-1]

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
