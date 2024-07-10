import time
import os
import numpy as np
import pandas as pd
from itertools import chain
from myutils.util import check_and_create_path, load_edge_ids, load_pkl
import torch
from runner.separated.base_runner import Runner
from runner.separated.trajectory_prediction import InformerModule, Trajectory_Loader


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.simulation_scenario = self.all_args.simulation_scenario
        self.time_range = self.all_args.time_range
        self.time_slots = 60  # 车辆移动过程中进行60个时间片(s)的任务，每个时间片一个任务

        # 加载server坐标
        edge_pos_file = "sumo/data/{}/edge_position.csv".format(self.simulation_scenario)
        self.edge_pos = pd.read_csv(edge_pos_file)

        # 轨迹预测模型
        self.seq_len = 32
        self.label_len = 16
        self.pred_len = 8
        self.TrajPredictionModel = InformerModule(enc_in=2, dec_in=2, c_out=2, seq_len=self.seq_len, label_len=self.label_len,
                                   pred_len=self.pred_len, factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1,
                                   d_ff=2048, dropout=0.05, attn='prob', embed='timeF', freq='s', activation='gelu',
                                   output_attention=False, distil=True, mix=True, device=torch.device('cuda:0'))
        setting = '{}_{}/seq{}label{}pre{}itr{}'.format(
            self.simulation_scenario, self.time_range, self.seq_len, self.label_len, self.pred_len, 0)
        self.TrajPredictionModel.load(setting)  # load model and scaler

        # 加载模拟数据 (初步使用模拟和预处理好的数据, 后续可改为实时调用的API)
        simulation_data_file = "sumo/data/{}/{}/simulation_sequences.pkl".format(self.simulation_scenario, self.time_range)
        # simulation_sequences --- key: start_time, value: one_sequence
        # one_sequence --- time_len=60长度的edge_vehicle_map --- key: edge_id, value: list of vehicle_id
        # {start_time: [{edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]},{},...,time_len],...,episode}
        self.simulation_sequences = load_pkl(simulation_data_file)
        self.sequence_keys = list(self.simulation_sequences.keys())  # list of start_time
        self.episode = len(self.sequence_keys)  # 迭代次数
        print("training episode: {}".format(self.episode))
        print(self.sequence_keys)

        # 加载车辆历史轨迹数据
        vehicle_trajectory_file = "st_prediction/trajectory_data/{}/{}/vehicle_trajectory.pkl".format(
            self.simulation_scenario, self.time_range)
        self.vehicle_trajectory = load_pkl(vehicle_trajectory_file)

        # 加载每个时刻系统内车辆数据
        time_vehicle_file = "st_prediction/trajectory_data/{}/{}/time_vehicle.pkl".format(
            self.simulation_scenario, self.time_range)
        self.time_vehicle = load_pkl(time_vehicle_file)

    def get_prediction_traj(self, start_time):
        # 进行预测
        current_vehicles = self.time_vehicle[start_time]  # start_time时刻系统内的车辆
        future_trajectory = {}  # key: vehicle_id, value: predict_trajectory (numpy)
        predict_vehicles = []  # 记录放入history_trajectories中的车的顺序
        seqs_x, seqs_x_mark, seqs_y, seqs_y_mark = [], [], [], []
        for vehicle in current_vehicles:
            time_and_trajectory = self.vehicle_trajectory[vehicle]
            trajectory_loader = Trajectory_Loader(time_and_trajectory)
            query_flag, seq_x, seq_x_mark, seq_y, seq_y_mark = trajectory_loader.query_history(
                start_time, self.seq_len, self.label_len, self.pred_len)
            if query_flag == 1:
                predict_vehicles.append(vehicle)
                seqs_x.append(seq_x)
                seqs_x_mark.append(seq_x_mark)
                seqs_y.append(seq_y)
                seqs_y_mark.append(seq_y_mark)
            else:
                future_trajectory[vehicle] = None
        if len(predict_vehicles) > 0:
            predict_edge_ids = self.TrajPredictionModel.predict(self.edge_pos,
                               np.array(seqs_x), np.array(seqs_x_mark), np.array(seqs_y), np.array(seqs_y_mark))
            for index, vehicle in enumerate(predict_vehicles):
                future_trajectory[vehicle] = predict_edge_ids[index]
        return future_trajectory

    @torch.no_grad()
    def collect_offloading(self, off_obs, off_share_obs):
        off_actions = {}
        off_actions_env = {}
        for agent_id in range(self.num_agents):
            self.mappotrainer[agent_id].prep_rollout()
            if agent_id in off_obs.keys():
                agent_off_obs = off_obs[agent_id]  # (vehicle num, 8)
                agent_off_share_obs = off_share_obs[agent_id]    # (vehicle num, 6+2*num_edge)
                if not self.use_centralized_Q1:  # default: True
                    agent_off_share_obs = off_obs[agent_id]

                agent_off_actions = []
                agent_off_actions_env = []
                for agent_off_ob, agent_off_share_ob in zip(agent_off_obs, agent_off_share_obs):
                    step = self.mappobuffer[agent_id].step
                    self.mappobuffer[agent_id].obs[step] = agent_off_ob.copy()
                    self.mappobuffer[agent_id].share_obs[step] = agent_off_share_ob.copy()
                    value, off_action, action_log_prob, rnn_state, rnn_state_critic = self.mappotrainer[agent_id].policy.get_actions(
                        self.mappobuffer[agent_id].share_obs[step],
                        self.mappobuffer[agent_id].obs[step],
                        self.mappobuffer[agent_id].rnn_states[step],
                        self.mappobuffer[agent_id].rnn_states_critic[step],
                        self.mappobuffer[agent_id].masks[step],
                    )
                    self.mappobuffer[agent_id].value_preds[step] = _t2n(value).copy()
                    self.mappobuffer[agent_id].actions[step] = _t2n(off_action).copy()
                    self.mappobuffer[agent_id].action_log_probs[step] = np.squeeze(_t2n(action_log_prob), axis=0).copy()
                    self.mappobuffer[agent_id].rnn_states[step + 1] = np.squeeze(_t2n(rnn_state), axis=0).copy()
                    self.mappobuffer[agent_id].rnn_states_critic[step + 1] = np.squeeze(_t2n(rnn_state_critic), axis=0).copy()

                    self.mappobuffer[agent_id].step = (step + 1) % self.mappobuffer[agent_id].buffer_length

                    off_action = _t2n(off_action)[0]
                    agent_off_actions.append(off_action)
                    action_env = np.eye(self.envs.action1_space[agent_id].n)[off_action]  # rearrange action
                    agent_off_actions_env.append(action_env)

                off_actions[agent_id] = np.array(agent_off_actions)
                off_actions_env[agent_id] = np.array(agent_off_actions_env)

        return off_actions, off_actions_env

    @torch.no_grad()
    def collect_allocation(self, off_obs, off_share_obs, off_actions_env):
        allo_actions = {}
        self.maddpgtrainer.prep_rollout()
        for agent_id in range(self.num_agents):
            if agent_id in off_obs.keys():
                agent_off_obs = off_obs[agent_id]  # (vehicle num, 8)
                agent_off_share_obs = off_share_obs[agent_id]  # (vehicle num, 6+2*num_edge)
                if not self.use_centralized_Q2:  # default: True
                    agent_off_share_obs = off_obs[agent_id]
                agent_allo_obs = np.hstack((off_actions_env[agent_id], agent_off_obs))
                agent_allo_share_obs = np.hstack((off_actions_env[agent_id], agent_off_share_obs))

                agent_allo_actions = []
                for agent_allo_ob, agent_allo_share_ob in zip(agent_allo_obs, agent_allo_share_obs):
                    if agent_allo_ob[self.num_agents] != 1:
                        step = self.maddpgbuffer[agent_id].step
                        self.maddpgbuffer[agent_id].obs[step] = agent_allo_ob.copy()
                        self.maddpgbuffer[agent_id].share_obs[step] = agent_allo_share_ob.copy()
                        allo_action, _ = self.maddpgtrainer.policies[agent_id].get_actions(
                            self.maddpgbuffer[agent_id].obs[step], explore=True)
                        self.maddpgbuffer[agent_id].actions[step] = _t2n(allo_action).copy()
                        self.maddpgbuffer[agent_id].step = (step + 1) % self.maddpgbuffer[agent_id].buffer_length

                        allo_action = _t2n(allo_action)
                        agent_allo_actions.append(allo_action)

                allo_actions[agent_id] = np.array(agent_allo_actions)
        return allo_actions

    def insert_mappo(self, counters, next_obs, next_share_obs, rewards):
        for agent_id in range(self.num_agents):
            if agent_id in next_obs.keys():
                agent_next_obs = next_obs[agent_id]
                agent_next_share_obs = next_share_obs[agent_id]
                if not self.use_centralized_Q1:  # default: True
                    agent_next_share_obs = next_obs[agent_id]
                for i in range(agent_next_obs.shape[0]):
                    step = counters[agent_id]
                    self.mappobuffer[agent_id].rewards[step] = rewards[agent_id][i]
                    self.mappobuffer[agent_id].next_obs[step] = agent_next_obs[i].copy()
                    self.mappobuffer[agent_id].next_share_obs[step] = agent_next_share_obs[i].copy()
                    counters[agent_id] = (step + 1) % self.mappobuffer[agent_id].buffer_length

    def insert_maddpg(self, counters, next_obs, next_share_obs, rewards, off_actions_env):
        for agent_id in range(self.num_agents):
            if agent_id in next_obs.keys():
                agent_next_obs = next_obs[agent_id]
                agent_next_share_obs = next_share_obs[agent_id]
                if not self.use_centralized_Q2:  # default: True
                    agent_next_share_obs = next_obs[agent_id]
                agent_next_obs = np.hstack((off_actions_env[agent_id], agent_next_obs))
                agent_next_share_obs = np.hstack((off_actions_env[agent_id], agent_next_share_obs))
                for i in range(agent_next_obs.shape[0]):
                    if agent_next_obs[i][self.num_agents] != 1:
                        step = counters[agent_id]
                        self.maddpgbuffer[agent_id].rewards[step] = rewards[agent_id][i]
                        self.maddpgbuffer[agent_id].next_obs[step] = agent_next_obs[i].copy()
                        self.maddpgbuffer[agent_id].next_share_obs[step] = agent_next_share_obs[i].copy()
                        counters[agent_id] = (step + 1) % self.maddpgbuffer[agent_id].buffer_length

    def run(self):
        print("Start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        metrics = self.clear_metrics()
        for epd in range(self.episode):  # episode步
            print('||                            Episode:{}                            ||'.format(epd))

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.mappotrainer[agent_id].policy.lr_decay(epd, self.episode)
                    self.maddpgtrainer.policies[agent_id].lr_decay(epd, self.episode)

            # record metrics
            epd_metrics = self.clear_epd_metrics()

            # 加载模拟数据
            start_time = self.sequence_keys[epd]
            one_sequence = self.simulation_sequences[start_time]

            all_task_num = 0  # 1个episode中任务总数
            # 1个episode中每个edge的任务总数--统计每个edge的流量
            edge_task_nums = [0] * self.num_agents
            # 1个episode中卸载/迁移到每个edge的任务数
            edge_comp_task_nums = [0] * self.num_agents
            counters_mappo = [0]*self.num_agents
            counters_maddpg = [0] * self.num_agents

            # 每个episode的time_slots步
            for current_step in range(self.time_slots-1):
                if current_step == 0:
                    # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
                    edge_vehicle_map = one_sequence[current_step]
                    future_trajectory = self.get_prediction_traj(start_time + current_step)
                    task_num_now, task_nums, off_obs, off_share_obs = self.envs.reset(edge_vehicle_map, future_trajectory)  # obs: dict
                    all_task_num += task_num_now
                    edge_task_nums = [edge_task_nums[i] + task_nums[i] for i in range(self.num_agents)]
                # sample offloading actions
                off_actions, off_actions_env = self.collect_offloading(off_obs, off_share_obs)
                # sample allocation actions
                allo_actions = self.collect_allocation(off_obs, off_share_obs, off_actions_env)
                # observe reward and next obs
                next_edge_vehicle_map = one_sequence[current_step+1]
                next_future_trajectory = self.get_prediction_traj(start_time + current_step + 1)
                if epd == 94:
                    print("94")
                task_num_now, task_nums, comp_task_nums, off_obs, off_share_obs, off_next_obs, off_next_share_obs, rewards, info, done = self.envs.step(
                    off_actions, allo_actions, edge_vehicle_map, next_edge_vehicle_map, next_future_trajectory)
                all_task_num += task_num_now
                edge_task_nums = [edge_task_nums[i] + task_nums[i] for i in range(self.num_agents)]
                edge_comp_task_nums = [edge_comp_task_nums[i] + comp_task_nums[i] for i in range(self.num_agents)]
                edge_vehicle_map = next_edge_vehicle_map
                # insert data into mappo buffer
                self.insert_mappo(counters_mappo, off_next_obs, off_next_share_obs, rewards)
                # insert data into maddpg buffer
                self.insert_maddpg(counters_maddpg, off_next_obs, off_next_share_obs, rewards, off_actions_env)
                # update metrics
                epd_metrics = self.update_epd_metrics(epd_metrics, rewards, info)

            # compute return and update network
            self.compute()
            mappo_train_infos = self.train_mappo()
            maddpg_train_infos = self.train_maddpg(epd)

            # save model
            if epd != 0 and (epd % self.save_interval == 0 or epd == self.episode - 1):
                self.save_mappo()
                self.save_maddpg()

            if epd % self.log_interval == 0:
                self.log_epd_metrics("train_mappo_maddpg_", epd_metrics, all_task_num, edge_task_nums, edge_comp_task_nums, epd)

            metrics = self.update_metrics(metrics, epd_metrics, all_task_num, edge_task_nums, edge_comp_task_nums)
            print("Complete time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        self.save_metrics(metrics)
        print("End time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    @torch.no_grad()
    def get_offloading_actions(self, off_obs, test_rnn_states):
        test_masks = np.ones((1, 1), dtype=np.float32)

        off_actions = {}
        off_actions_env = {}
        for agent_id in range(self.num_agents):
            self.mappotrainer[agent_id].prep_rollout()
            if agent_id in off_obs.keys():
                agent_off_obs = off_obs[agent_id]  # (vehicle num, 8)
                agent_off_actions = []
                agent_off_actions_env = []
                for agent_off_ob in agent_off_obs:
                    off_action, test_rnn_states = self.mappotrainer[agent_id].policy.act(
                        agent_off_ob, test_rnn_states, test_masks, deterministic=True)
                    off_action = _t2n(off_action)[0]
                    agent_off_actions.append(off_action)
                    action_env = np.eye(self.envs.action1_space[agent_id].n)[off_action]  # rearrange action
                    agent_off_actions_env.append(action_env)
                    test_rnn_states = _t2n(test_rnn_states)
                off_actions[agent_id] = np.array(agent_off_actions)
                off_actions_env[agent_id] = np.array(agent_off_actions_env)

        return off_actions, off_actions_env, test_rnn_states

    @torch.no_grad()
    def get_allocation_actions(self, off_obs, off_actions_env):
        allo_actions = {}
        self.maddpgtrainer.prep_rollout()
        for agent_id in range(self.num_agents):
            if agent_id in off_obs.keys():
                agent_off_obs = off_obs[agent_id]  # (vehicle num, 8)
                agent_allo_obs = np.hstack((off_actions_env[agent_id], agent_off_obs))
                agent_allo_actions = []
                for agent_allo_ob in agent_allo_obs:
                    if agent_allo_ob[self.num_agents] != 1:
                        allo_action, _ = self.maddpgtrainer.policies[agent_id].get_actions(
                            agent_allo_ob, explore=False)
                        allo_action = _t2n(allo_action)
                        agent_allo_actions.append(allo_action)
                allo_actions[agent_id] = np.array(agent_allo_actions)
        return allo_actions

    @torch.no_grad()
    def test(self, start_epi, end_epi):
        print("Start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        metrics = self.clear_metrics()
        test_rnn_states = np.zeros((1, self.recurrent_N, self.hidden_size1), dtype=np.float32)
        for epd in range(start_epi, end_epi):  # episode步
            print('||                            Episode:{}                            ||'.format(epd))

            # record metrics
            epd_metrics = self.clear_epd_metrics()

            # 加载模拟数据
            start_time = self.sequence_keys[epd]
            one_sequence = self.simulation_sequences[start_time]

            all_task_num = 0  # 1个episode中任务总数
            # 1个episode中每个edge的任务总数--统计每个edge的流量
            edge_task_nums = [0] * self.num_agents
            # 1个episode中卸载/迁移到每个edge的任务数
            edge_comp_task_nums = [0] * self.num_agents

            # 每个episode的time_slots步
            for current_step in range(self.time_slots - 1):
                if current_step == 0:
                    # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
                    edge_vehicle_map = one_sequence[current_step]
                    future_trajectory = self.get_prediction_traj(start_time + current_step)
                    task_num_now, task_nums, off_obs, _ = self.envs.reset(edge_vehicle_map, future_trajectory)   # obs: dict
                    all_task_num += task_num_now
                    edge_task_nums = [edge_task_nums[i] + task_nums[i] for i in range(self.num_agents)]
                # get offloading actions
                off_actions, off_actions_env, test_rnn_states = self.get_offloading_actions(off_obs, test_rnn_states)
                # get allocation actions
                allo_actions = self.get_allocation_actions(off_obs, off_actions_env)
                # observe reward and next obs
                next_edge_vehicle_map = one_sequence[current_step + 1]
                next_future_trajectory = self.get_prediction_traj(start_time + current_step + 1)
                task_num_now, task_nums, comp_task_nums, off_obs, _, _, _, rewards, info, done = self.envs.step(
                    off_actions, allo_actions, edge_vehicle_map, next_edge_vehicle_map, next_future_trajectory)
                all_task_num += task_num_now
                edge_task_nums = [edge_task_nums[i] + task_nums[i] for i in range(self.num_agents)]
                edge_comp_task_nums = [edge_comp_task_nums[i] + comp_task_nums[i] for i in range(self.num_agents)]
                edge_vehicle_map = next_edge_vehicle_map
                # update metrics
                epd_metrics = self.update_epd_metrics(epd_metrics, rewards, info)

            if epd % self.log_interval == 0:
                self.log_epd_metrics("test_mappo_maddpg_", epd_metrics, all_task_num, edge_task_nums, edge_comp_task_nums, epd)
            metrics = self.update_metrics(metrics, epd_metrics, all_task_num, edge_task_nums, edge_comp_task_nums)
            print("Complete time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        self.save_metrics(metrics)

        _, _, _, avr_per_task, avr_comp_per_task, avr_comm_per_task, failure_rate, local_rate, \
        edge_util_per_time, vehicle_util_per_time, _ = metrics
        print("Test average latency per task: {}, Test average computation latency per task:{},"
              "Test average communication latency per task: {}".format(
            np.mean(avr_per_task), np.mean(avr_comp_per_task), np.mean(avr_comm_per_task)))
        print("Test average task failure rate: {}, Test average local task rate:{} ".format(
            np.mean(failure_rate), np.mean(local_rate)))
        print("Test average edge util per time: {}, Test average vehicle util per time:{} ".format(
            np.mean(edge_util_per_time), np.mean(vehicle_util_per_time)))
        print("End time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
