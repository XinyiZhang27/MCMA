import numpy as np
from myutils.util import get_dim_from_space


class MlpPolicyBuffer(object):

    def __init__(self, args, obs_space, share_obs_space, act_space):

        self.buffer_length = args.buffer_length2
        self.use_reward_normalization = args.use_reward_normalization

        obs_dim = get_dim_from_space(obs_space)
        share_obs_dim = get_dim_from_space(share_obs_space)
        act_dim = get_dim_from_space(act_space)

        self.obs = np.zeros((self.buffer_length, obs_dim), dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_length, share_obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs, dtype=np.float32)
        self.next_share_obs = np.zeros_like(self.share_obs, dtype=np.float32)

        self.actions = np.zeros((self.buffer_length, act_dim), dtype=np.float32)
        self.avail_acts = np.ones_like(self.actions, dtype=np.float32)
        self.next_avail_acts = np.ones_like(self.avail_acts, dtype=np.float32)

        self.rewards = np.zeros((self.buffer_length, 1), dtype=np.float32)

        self.step = 0

    def compute_next_acts_Qs(self):
        for step in reversed(range(self.step)):

                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * \
                        self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])

        next_acts, _ = self.policies[agent_id].get_actions(next_obs, next_avail_acts, use_target=True)
        next_Qs = self.policies[agent_id].target_critic(next_share_obs, next_acts)

    def sample(self, batch_size):
        size = min(self.step, batch_size)
        sample_inds = np.random.choice(self.step, size=size)

        obs = self.obs[sample_inds]
        share_obs = self.share_obs[sample_inds]
        next_obs = self.next_obs[sample_inds]
        next_share_obs = self.next_share_obs[sample_inds]

        actions = self.actions[sample_inds]
        avail_acts = self.avail_acts[sample_inds]
        next_avail_acts = self.next_avail_acts[sample_inds]

        if self.use_reward_normalization:
            mean_reward = self.rewards[:self.step].mean()
            std_reward = self.rewards[:self.step].std()
            rewards = (self.rewards[sample_inds] - mean_reward) / std_reward
        else:
            rewards = self.rewards[sample_inds]

        return obs, share_obs, actions, rewards, next_obs, next_share_obs, avail_acts, next_avail_acts
