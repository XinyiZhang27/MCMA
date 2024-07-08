import time
import os
import numpy as np
from itertools import chain
import torch
from torch.utils.tensorboard import SummaryWriter
from myutils.util import save_pkl
from myutils.mappo_utils.separated_buffer import SeparatedReplayBuffer
from myutils.maddpg_utils.mlp_buffer import MlpPolicyBuffer
from algorithms.MAPPO.mappo import MAPPO
from algorithms.MAPPO.MAPPOPolicy import MAPPOPolicy
from algorithms.MADDPG.maddpg import MADDPG
from algorithms.MADDPG.MADDPGPolicy import MADDPGPolicy


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay

        # mappo
        self.algorithm_name1 = self.all_args.algorithm_name1
        self.use_centralized_Q1 = self.all_args.use_centralized_Q1
        self.hidden_size1 = self.all_args.hidden_size1
        self.recurrent_N = self.all_args.recurrent_N

        # maddpg
        self.use_centralized_Q2 = self.all_args.use_centralized_Q2
        self.batch_size = self.all_args.batch_size
        self.use_soft_update = self.all_args.use_soft_update
        self.hard_update_interval = self.all_args.hard_update_interval

        # dir
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        self.metrics_dir = str(self.run_dir / "metrics")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
        if self.all_args.stage == "train":
            self.save_dir1 = str(self.run_dir / "models" / "MAPPO")
            if not os.path.exists(self.save_dir1):
                os.makedirs(self.save_dir1)
            self.save_dir2 = str(self.run_dir / "models" / "MADDPG")
            if not os.path.exists(self.save_dir2):
                os.makedirs(self.save_dir2)

        # policies
        self.mappopolicy = []
        self.maddpgpolicy = []
        for agent_id in range(self.num_agents):
            share_observation1_space = (
                self.envs.share_observation1_space[agent_id]
                if self.use_centralized_Q1
                else self.envs.observation1_space[agent_id]
            )
            # MAPPO policy network
            po1 = MAPPOPolicy(
                self.all_args,
                self.envs.observation1_space[agent_id],
                share_observation1_space,
                self.envs.action1_space[agent_id],
                device=self.device,
            )
            self.mappopolicy.append(po1)

            share_observation2_space = (
                self.envs.share_observation2_space[agent_id]
                if self.use_centralized_Q2
                else self.envs.observation2_space[agent_id]
            )
            # MADDPG policy network
            po2 = MADDPGPolicy(
                self.all_args,
                self.envs.observation2_space[agent_id],
                share_observation2_space,
                self.envs.action2_space[agent_id],
                device=self.device,
            )
            self.maddpgpolicy.append(po2)

        if self.all_args.stage == "test":
            if self.all_args.model_dir1 is not None:
                self.restore(self.mappopolicy, str(self.all_args.model_dir1 / "models" / "MAPPO"))
            if self.all_args.model_dir2 is not None:
                self.restore(self.maddpgpolicy, str(self.all_args.model_dir2 / "models" / "MADDPG"))

        self.mappotrainer = []
        self.mappobuffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr1 = MAPPO(self.all_args, self.mappopolicy[agent_id], device=self.device)
            # buffer
            share_observation1_space = (
                self.envs.share_observation1_space[agent_id]
                if self.use_centralized_Q1
                else self.envs.observation1_space[agent_id]
            )
            bu1 = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation1_space[agent_id],
                share_observation1_space,
                self.envs.action1_space[agent_id],
            )
            self.mappotrainer.append(tr1)
            self.mappobuffer.append(bu1)

        self.maddpgtrainer = MADDPG(self.all_args, self.num_agents, self.maddpgpolicy, device=self.device)
        self.maddpgbuffer = []
        for agent_id in range(self.num_agents):
            # buffer
            share_observation2_space = (
                self.envs.share_observation2_space[agent_id]
                if self.use_centralized_Q2
                else self.envs.observation2_space[agent_id]
            )
            bu2 = MlpPolicyBuffer(
                self.all_args,
                self.envs.observation2_space[agent_id],
                share_observation2_space,
                self.envs.action2_space[agent_id],
            )
            self.maddpgbuffer.append(bu2)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            step = self.mappobuffer[agent_id].step
            if step > 0:
                self.mappotrainer[agent_id].prep_rollout()
                next_value = self.mappotrainer[agent_id].policy.get_values(
                    self.mappobuffer[agent_id].next_share_obs[step-1],
                    self.mappobuffer[agent_id].rnn_states_critic[step],
                    self.mappobuffer[agent_id].masks[step],
                )
                next_value = _t2n(next_value)
                self.mappobuffer[agent_id].compute_returns(next_value, self.mappotrainer[agent_id].value_normalizer)

    def train_mappo(self):
        train_infos = {}
        for agent_id in range(self.num_agents):
            if self.mappobuffer[agent_id].step > 0:
                self.mappotrainer[agent_id].prep_training()
                train_info = self.mappotrainer[agent_id].train(self.mappobuffer[agent_id])
                train_infos[agent_id] = train_info
                self.mappobuffer[agent_id].after_update()
        return train_infos

    def train_maddpg(self, epd):
        train_infos = {}
        self.maddpgtrainer.prep_training()
        # gradient updates
        for agent_id in range(self.num_agents):
            if self.maddpgbuffer[agent_id].step > self.batch_size:
                sample = self.maddpgbuffer[agent_id].sample(self.batch_size)
                train_info = self.maddpgtrainer.train(agent_id, sample)
                train_infos[agent_id] = train_info
            if self.use_soft_update:
                self.maddpgpolicy[agent_id].soft_target_updates()
            elif epd != 0 and epd % self.hard_update_interval == 0:
                self.maddpgpolicy[agent_id].hard_target_updates()
        return train_infos

    def save_mappo(self):
        for agent_id in range(self.num_agents):
            # mappo
            mappo_actor = self.mappotrainer[agent_id].policy.actor
            torch.save(mappo_actor.state_dict(), str(self.save_dir1) + "/actor_agent" + str(agent_id) + ".pt")
            mappo_critic = self.mappotrainer[agent_id].policy.critic
            torch.save(mappo_critic.state_dict(), str(self.save_dir1) + "/critic_agent" + str(agent_id) + ".pt")

    def save_maddpg(self):
        for agent_id in range(self.num_agents):
            # maddpg
            maddpg_actor = self.maddpgtrainer.policies[agent_id].actor
            torch.save(maddpg_actor.state_dict(), str(self.save_dir2) + "/actor_agent" + str(agent_id) + ".pt")
            maddpg_critic = self.maddpgtrainer.policies[agent_id].critic
            torch.save(maddpg_critic.state_dict(), str(self.save_dir2) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self, policy, model_dir):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + str(agent_id) + ".pt")
            policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_epd_metrics(self, tag, epd_metrics, all_task_num, edge_task_nums, edge_comp_task_nums, epd):
        edge_utils, vehicle_utils, all_rewards, all_latency, all_wait_latency, all_exe_latency,\
        all_tran_latency, all_mig_latency, all_failure_num, all_local_num = epd_metrics
        self.writer.add_scalars(tag+"total_task_num", {"total_task_num": all_task_num}, epd)
        for i in range(self.num_agents):
            self.writer.add_scalars(tag + "task_num", {"edge%i_task_num" % i: edge_task_nums[i]}, epd)
            self.writer.add_scalars(tag + "comp_task_num", {"edge%i_comp_task_num" % i: edge_comp_task_nums[i]}, epd)
        self.writer.add_scalars(tag+"latency", {"average_completion_latency_per_task": all_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_wait_latency_per_task": all_wait_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_exe_latency_per_task": all_exe_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_tran_latency_per_task": all_tran_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_mig_latency_per_task": all_mig_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"rate", {"task_failure_rate": all_failure_num / all_task_num}, epd)
        self.writer.add_scalars(tag+"rate", {"local_task_rate": all_local_num / all_task_num}, epd)
        self.writer.add_scalars(tag+"ratio", {"average_utilization_of_edge_computation_resources": np.mean(edge_utils)}, epd)
        self.writer.add_scalars(tag+"ratio", {"average_utilization_of_vehicle_computation_resources": np.mean(vehicle_utils)}, epd)
        self.writer.add_scalars(tag+"reward", {"average_reward_per_task": all_rewards / all_task_num}, epd)

        print('average completion latency per task: %.4f, average computation latency per task: %.4f,'
              'average communication latency per task: %.4f,task failure rate: %.4f, local task rate: %.4f' % (
                all_latency / all_task_num, (all_wait_latency+all_exe_latency) / all_task_num,
                (all_tran_latency+all_mig_latency) / all_task_num, all_failure_num / all_task_num, all_local_num / all_task_num))
        print('average utilization of edge computation resources: %.4f' % (np.mean(edge_utils)))
        print('average utilization of vehicle computation resources: %.4f' % (np.mean(vehicle_utils)))

    def clear_epd_metrics(self):
        edge_utils = []  # 1个episode中time_slots步每一步的平均edge利用率
        vehicle_utils = []  # 1个episode中time_slots步每一步的平均vehicle利用率
        all_rewards = 0  # 1个episode中time_slots步所有edges的总rewards
        all_latency = 0  # 1个episode中time_slots步所有edges的总时延
        all_wait_latency = 0  # 1个episode中time_slots步所有edges的等待时延
        all_exe_latency = 0
        all_tran_latency = 0
        all_mig_latency = 0
        all_failure_num = 0  # 1个episode中time_slots步所有edges的总失败任务数
        all_local_num = 0  # 1个episode中time_slots步所有edges的总本地执行任务数
        return edge_utils, vehicle_utils, all_rewards, all_latency, all_wait_latency, all_exe_latency, \
               all_tran_latency, all_mig_latency, all_failure_num, all_local_num

    def update_epd_metrics(self, epd_metrics, rewards, info):
        edge_utils, vehicle_utils, all_rewards, all_latency, all_wait_latency, all_exe_latency, \
        all_tran_latency, all_mig_latency, all_failure_num, all_local_num = epd_metrics
        edge_utils.append(info["average_edge_utilization"])
        vehicle_utils.append(info["average_vehicle_utilization"])
        all_rewards += np.sum([np.sum(x) for x in rewards.values()])  # 加上所有edges的总rewards
        all_latency += np.sum([x for x in info["latencies"].values()])  # 加上所有edges的时延
        all_wait_latency += np.sum([x for x in info["wait_latencies"].values()])  # 加上所有edges的等待时延
        all_exe_latency += np.sum([x for x in info["exe_latencies"].values()])
        all_tran_latency += np.sum([x for x in info["tran_latencies"].values()])
        all_mig_latency += np.sum([x for x in info["mig_latencies"].values()])
        all_failure_num += np.sum([x for x in info["failure_nums"].values()])  # 加上所有edges的失败任务数
        all_local_num += np.sum([x for x in info["local_nums"].values()])
        return edge_utils, vehicle_utils, all_rewards, all_latency, all_wait_latency, all_exe_latency, \
               all_tran_latency, all_mig_latency, all_failure_num, all_local_num

    def clear_metrics(self):
        all_task_num, edge_task_nums, edge_comp_task_nums, \
        avr_per_task, avr_comp_per_task, avr_comm_per_task, failure_rate, local_rate, \
        edge_util_per_time, vehicle_util_per_time, avr_reward = [], [], [], [], [], [], [], [], [], [], []
        return all_task_num, edge_task_nums, edge_comp_task_nums, \
               avr_per_task, avr_comp_per_task, avr_comm_per_task, failure_rate, local_rate, \
               edge_util_per_time, vehicle_util_per_time, avr_reward

    def update_metrics(self, metrics, epd_metrics, all_task_num, edge_task_nums, edge_comp_task_nums):
        overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, \
        avr_per_task, avr_comp_per_task, avr_comm_per_task, failure_rate, local_rate, \
        edge_util_per_time, vehicle_util_per_time, avr_reward = metrics
        edge_utils, vehicle_utils, all_rewards, all_latency, all_wait_latency, all_exe_latency, \
        all_tran_latency, all_mig_latency, all_failure_num, all_local_num = epd_metrics

        overall_task_num.append(all_task_num)
        overall_edge_task_nums.append(edge_task_nums)
        overall_edge_comp_task_nums.append(edge_comp_task_nums)
        avr_per_task.append(all_latency / all_task_num)
        avr_comp_per_task.append((all_wait_latency + all_exe_latency) / all_task_num)
        avr_comm_per_task.append((all_tran_latency + all_mig_latency) / all_task_num)
        failure_rate.append(all_failure_num / all_task_num)
        local_rate.append(all_local_num / all_task_num)
        edge_util_per_time.append(np.mean(edge_utils))
        vehicle_util_per_time.append(np.mean(vehicle_utils))
        avr_reward.append(all_rewards / all_task_num)
        return overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, \
               avr_per_task, avr_comp_per_task, avr_comm_per_task, failure_rate, local_rate, \
               edge_util_per_time, vehicle_util_per_time, avr_reward

    def save_metrics(self, metrics):
        overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, \
        avr_per_task, avr_comp_per_task, avr_comm_per_task, failure_rate, local_rate, \
        edge_util_per_time, vehicle_util_per_time, avr_reward = metrics

        save_pkl(self.metrics_dir + "/overall_task_num.pkl", overall_task_num)
        save_pkl(self.metrics_dir + "/overall_edge_task_nums.pkl", overall_edge_task_nums)
        save_pkl(self.metrics_dir + "/overall_edge_comp_task_nums.pkl", overall_edge_comp_task_nums)
        save_pkl(self.metrics_dir + "/latency.pkl", avr_per_task)
        save_pkl(self.metrics_dir + "/computation_latency.pkl", avr_comp_per_task)
        save_pkl(self.metrics_dir + "/communication_latency.pkl", avr_comm_per_task)
        save_pkl(self.metrics_dir + "/failure_rate.pkl", failure_rate)
        save_pkl(self.metrics_dir + "/local_rate.pkl", local_rate)
        save_pkl(self.metrics_dir + "/edge_util.pkl", edge_util_per_time)
        save_pkl(self.metrics_dir + "/vehicle_util.pkl", vehicle_util_per_time)
        save_pkl(self.metrics_dir + "/reward.pkl", avr_reward)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writer.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writer.add_scalars(k, {k: np.mean(v)}, total_num_steps)
