import torch
import numpy as np
import copy
import itertools
from myutils.util import huber_loss, mse_loss, check
from myutils.maddpg_utils.popart import PopArt

class MADDPG:
    def __init__(self, args, num_agents, policies, device=None):
        """
        Trainer class for MADDPG. See parent class for more information.
        :param actor_update_interval: (int) number of critic updates to perform between every update to the actor.
        """
        self.args = args
        self.use_popart = self.args.use_popart2
        self.use_huber_loss = self.args.use_huber_loss2
        self.huber_delta = self.args.huber_delta2
        self.gamma = self.args.gamma2
        self.max_grad_norm = self.args.max_grad_norm2
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = num_agents
        self.policies = policies
        if self.use_popart:
            self.value_normalizer = {i: PopArt(1) for i in range(num_agents)}
        self.num_updates = {i: 0 for i in range(num_agents)}

    def train(self, agent_id, batch):
        train_info = {}
        obs, share_obs, actions, rewards, next_obs, next_share_obs, avail_acts, next_avail_acts = batch

        # update critic
        next_acts, _ = self.policies[agent_id].get_actions(next_obs, next_avail_acts, use_target=True)
        next_Qs = self.policies[agent_id].target_critic(next_share_obs, next_acts)

        rewards = check(rewards).to(**self.tpdv)
        if self.use_popart:
            target_Qs = rewards + self.gamma * self.value_normalizer[agent_id].denormalize(next_Qs)
            target_Qs = self.value_normalizer[agent_id](target_Qs)
        else:
            target_Qs = rewards + self.gamma * next_Qs
        predicted_Qs = self.policies[agent_id].critic(share_obs, actions)
        error = target_Qs.detach() - predicted_Qs
        if self.use_huber_loss:
            critic_loss = huber_loss(error, self.huber_delta).mean()
        else:
            critic_loss = mse_loss(error).mean()

        self.policies[agent_id].critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policies[agent_id].critic_optimizer.step()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.policies[agent_id].critic.parameters(), self.max_grad_norm)

        train_info['critic_loss'] = critic_loss
        train_info['critic_grad_norm'] = critic_grad_norm

        # update actor
        # freeze Q-networks
        for p in self.policies[agent_id].critic.parameters():
            p.requires_grad = False

        acts, _ = self.policies[agent_id].get_actions(obs, avail_acts, use_gumbel=True)
        actor_loss = -self.policies[agent_id].critic(share_obs, acts).mean()

        self.policies[agent_id].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.policies[agent_id].actor_optimizer.step()

        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.policies[agent_id].actor.parameters(), self.max_grad_norm)

        for p in self.policies[agent_id].critic.parameters():
            p.requires_grad = True

        train_info['actor_loss'] = actor_loss
        train_info['actor_grad_norm'] = actor_grad_norm

        return train_info

    def prep_training(self):
        """See parent class."""
        for policy in self.policies:
            policy.actor.train()
            policy.critic.train()
            policy.target_actor.train()
            policy.target_critic.train()

    def prep_rollout(self):
        """See parent class."""
        for policy in self.policies:
            policy.actor.eval()
            policy.critic.eval()
            policy.target_actor.eval()
            policy.target_critic.eval()
