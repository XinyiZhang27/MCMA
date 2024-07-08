import numpy as np

class Edge:
    def __init__(self, edge_id, num_edge, cpu=12, bw=10):

        self.edge_id = edge_id
        self.num_edge = num_edge

        self.default_settings = {"cpu": cpu, "bw": bw}
        # edge 状态(3)
        self.cpu = cpu   # edge server 算力
        self.bw = bw     # edge server 带宽
        self.buff_CPU = 0  # 任务量队列(以所需CPU时间为单位)

        # 信道噪声建模
        self.noise = 2 / np.power(10, 13)

        # edge之间的数据传输速率
        self.trans_rate = 50

        # 覆盖车辆
        self.connected_vehicles = []

        # 预测值
        self.predict_trajectory = []

    def reset(self):
        self.cpu = self.default_settings["cpu"]  # edge server 算力
        self.bw = self.default_settings["bw"]  # edge server 带宽
        self.buff_CPU = 0  # 任务量队列(以所需CPU时间为单位)
        self.connected_vehicles = []  # 清空连接车辆
        self.predict_trajectory = []  # 重置预测值

    def vehicle_connection(self, vehicles):
        self.connected_vehicles = vehicles

    def clear_futuretrajectory(self):
        self.predict_trajectory = []

    def update_futuretrajectory(self, trajectory):
        self.predict_trajectory.append(trajectory)

    def get_current_task_sum(self, tasks):
        sum = 0
        for task_i in tasks:
            sum += task_i[0]  # data size
        return sum


class Task:
    def __init__(self, data_size, CPU_cycle, ddl):
        # 任务状态(3)
        self.data_size = data_size
        self.CPU_cycle = CPU_cycle
        self.ddl = ddl

        self.vehicle_id = -1


class Task_pool:
    def __init__(self):
        # (data_size, CPU_cycle, DDl)
        self.pool = [[90, 3, 5], [60, 3, 3], [90, 5, 3], [60, 5, 5]]

    def random_sample_n(self, size):
        task_list = []
        sampled_index = np.random.choice([0, 1, 2, 3], size=size, p=[0.25, 0.25, 0.25, 0.25])
        for i in sampled_index:
            task_i = self.pool[i]
            one_task = Task(task_i[0], task_i[1], task_i[2])
            task_list.append(one_task)
        return task_list


class Vehicle:
    def __init__(self, vehicle_id, cpu=2):

        self.vehicle_id = vehicle_id

        # vehicle状态(3)
        self.cpu = cpu   # 算力
        self.buff_CPU = 0  # 任务量队列(以所需CPU时间为单位)
        self.trans = 0.2  # 传输功率
        self.channel_gain = 50  #

        # edge 连接
        self.connected_edge = -1
        self.distance = -1  # 暂时不考虑

        # 当前任务信息
        self.current_task = -1

    def edge_connection(self, edge_id, distance=-1):
        self.connected_edge = edge_id
        self.distance = distance

    def generate_task(self, task_pool):
        one_task = task_pool.random_sample_n(size=1)[0]
        self.current_task = one_task


class EnvCore(object):
    def __init__(self, Edge_IDs, edge_index_map, shortest_paths, num_edge, time_slot=60, pred_len=8):
        # 设置edge
        self.edges = {}
        for i in Edge_IDs:
            edge_i = Edge(i, num_edge, cpu=12, bw=10)
            self.edges[i] = edge_i
        self.Edge_IDs = Edge_IDs
        self.Edge_index_map = edge_index_map
        self.shortest_paths = shortest_paths
        self.num_edge = num_edge
        self.time_slots = time_slot  # 强化学习评估时长（一分钟）
        self.pred_len = pred_len  # 预测轨迹长度

        # 任务建模
        self.task_pool = Task_pool()

        # 车辆建模
        self.vehicle_pool = {}  # 系统内当前车辆，vehicle_ID 为 key

        # 卸载 MAPPO
        self.obs1_dim = 6 + self.pred_len + 3
        self.action1_dim = self.num_edge + 1  # local + offload + migration
        self.share_obs1_dim = 6 + self.pred_len + self.num_edge*3

        # 分配 MADDPG
        self.obs2_dim = 6 + self.pred_len + (self.num_edge + 1) + 3
        self.action2_dim = 2  # bandwidth computation
        self.share_obs2_dim = 6 + self.pred_len + (self.num_edge + 1) + self.num_edge * 3

    def reset(self, edge_vehicle_map, future_trajectory):
        all_task_num = 0
        # 每个edge覆盖的车辆数--统计每个edge的流量
        edge_task_nums = [0] * self.num_edge
        # 重置系统内当前车辆
        self.vehicle_pool = {}  # 尚未考虑多个连续episode之间vehicle_pool的保留
        # 重置每个edge的建模
        for id_i, edge_i in self.edges.items():
            edge_i.reset()

        obs = {}
        share_obs = {}

        whole_edge_state = []
        for id_i, edge_i in self.edges.items():
            whole_edge_state.append([edge_i.cpu, edge_i.buff_CPU, edge_i.bw])
        whole_edge_state = np.array(whole_edge_state).flatten()

        # 对于每一个edge
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            # 车辆根据模拟数据加入vehicle_pool并与Edges连接，更新连接车辆的轨迹预测值
            # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
            if id_i in edge_vehicle_map:
                vehicle_ids = edge_vehicle_map[id_i]
                for v_id in vehicle_ids:
                    vehicle_i = Vehicle(v_id)  # 新建一个车辆
                    self.vehicle_pool[v_id] = vehicle_i  # 加入系统的车辆pool中
                    if future_trajectory[v_id] != None:
                        edge_i.update_futuretrajectory(
                            np.array([self.Edge_index_map[str(int(item)) if isinstance(item, float) else item]
                                      for item in future_trajectory[v_id]]))  # 更新预测值
                    else:
                        edge_i.update_futuretrajectory(np.full(self.pred_len, -1))
                all_task_num += len(vehicle_ids)
                edge_task_nums[agent_id] = len(vehicle_ids)  # 统计每个edge的流量
                edge_i.vehicle_connection(vehicle_ids)  # ID 记录至对应edge

            # 每个车辆生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # Edge Information
            edge_state = np.array([edge_i.cpu, edge_i.buff_CPU, edge_i.bw])

            if len(edge_i.connected_vehicles) > 0:
                obs[agent_id] = []
                share_obs[agent_id] = []
                # Vehicle information + Task information
                for i, v_id in enumerate(edge_i.connected_vehicles):
                    v_i = self.vehicle_pool[v_id]
                    # [车辆算力，车辆任务量队列(CPU)，传输功率；任务数据量，任务所需CPU，任务截止时间]
                    vehicle_task_state = [v_i.cpu, v_i.buff_CPU, v_i.trans,
                                          v_i.current_task.data_size, v_i.current_task.CPU_cycle, v_i.current_task.ddl]
                    prediction_state = edge_i.predict_trajectory[i]
                    obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, edge_state)))
                    share_obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, whole_edge_state)))

                obs[agent_id] = np.array(obs[agent_id])
                share_obs[agent_id] = np.array(share_obs[agent_id])

        self.done = False  # 模拟终止flag
        self.count_step = 0

        return all_task_num, edge_task_nums, obs, share_obs

    def step(self, off_actions, allo_actions, edge_vehicle_map, next_edge_vehicle_map, next_future_trajectory):
        latencies = {}  # 所有edges的总时延
        wait_latencies = {}  # 所有edges的等待时延
        tran_latencies = {}
        exe_latencies = {}
        mig_latencies = {}
        rewards = {}  # 所有edges的rewards
        failure_nums = {}  # 所有edges的失败数
        local_nums = {}  # 所有edges的本地执行任务数

        vehicle_resource_utilization = []  # 每一辆车
        edge_resource_utilization = []  # 每个edge

        # 卸载/迁移到每个edge的任务数--用于显示负载均衡
        edge_comp_task_nums = [0] * self.num_edge

        rearrange_cpus = {}  # 按照执行任务的edge重新划分cpu比例
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            if agent_id in off_actions.keys():
                allo_i = 0
                for off_i, off_action in enumerate(off_actions[agent_id]):
                    if off_action == self.num_edge:
                        continue
                    if off_action not in rearrange_cpus.keys():
                        rearrange_cpus[off_action] = []
                    rearrange_cpus[off_action].append(allo_actions[agent_id][allo_i][1])  # cpu比例
                    allo_i += 1

        # 统计所有需要卸载到每个edge的任务总量(CPU cycles), 后面统一更新每个edge_i.buff_CPU
        all_offload_CPU = {key: 0 for key in self.Edge_index_map.keys()}

        # 对于每一个edge
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            if agent_id in off_actions.keys():

                all_wait_latency = 0  # edge_i所有车辆的等待时延
                all_exe_latency = 0  # edge_i所有车辆的计算时延
                all_tran_latency = 0  # edge_i所有车辆的传输时延
                all_mig_latency = 0  # edge_i所有车辆的传输时延
                all_latency = 0  # edge_i所有车辆的总时延
                failure_num = 0  # edge_i所有车辆任务中失败的个数
                local_num = 0  # edge_i所有车辆任务中本地执行的个数
                agent_rewards = []

                off_action = off_actions[agent_id]
                allo_action = allo_actions[agent_id]
                index = 0

                for i, v_id in enumerate(edge_i.connected_vehicles):  # 对于每一辆车执行决策
                    vehicle_i = self.vehicle_pool[v_id]
                    # 得到bw和cpu的分配比例
                    if off_action[i] != self.num_edge:
                        rearrange_cpu = np.array(rearrange_cpus[off_action[i]])
                        ratio_bw = np.exp(allo_action[index][0]) / np.sum(np.exp(allo_action[:][0]))
                        ratio_cpu = np.exp(allo_action[index][1]) / np.sum(np.exp(rearrange_cpu[:]))
                        index += 1

                    # 不同卸载策略
                    if off_action[i] == agent_id:  # 边缘卸载
                        data_rate = (edge_i.bw * ratio_bw) * np.log2(1 + vehicle_i.trans * vehicle_i.channel_gain / edge_i.noise)
                        tran_latency = vehicle_i.current_task.data_size / data_rate  # 传输时延与任务量有关
                        exe_latency = vehicle_i.current_task.CPU_cycle / (edge_i.cpu * ratio_cpu)  # 执行时延与edge算力有关
                        wait_latency = 0
                        edge_comp_task_nums[agent_id] += 1
                        if edge_i.buff_CPU > 0:  # 边缘计算是否等待
                            wait_latency = edge_i.buff_CPU / edge_i.cpu
                        mig_latency = 0
                        latency = tran_latency + exe_latency + wait_latency + mig_latency
                        # 统计所有需要卸载到当前edge的任务总量(CPU cycles)
                        all_offload_CPU[id_i] += vehicle_i.current_task.CPU_cycle

                    elif off_action[i] == self.num_edge:  # 本地执行
                        tran_latency = 0
                        exe_latency = vehicle_i.current_task.CPU_cycle / vehicle_i.cpu
                        wait_latency = 0
                        if vehicle_i.buff_CPU > 0:  # 本地计算是否等待
                            wait_latency = vehicle_i.buff_CPU / vehicle_i.cpu
                        mig_latency = 0
                        latency = tran_latency + exe_latency + wait_latency + mig_latency  # 总时延
                        local_num += 1
                        # 更新车辆buff
                        vehicle_i.buff_CPU += vehicle_i.current_task.CPU_cycle

                    else:  # 任务迁移
                        next_edge_id = [key for key, value in self.Edge_index_map.items() if value == off_action[i]][0]
                        next_edge_i = self.edges[next_edge_id]
                        next_agent_id = self.Edge_index_map[next_edge_id]
                        data_rate = (next_edge_i.bw * ratio_bw) * np.log2(1 + vehicle_i.trans * vehicle_i.channel_gain / edge_i.noise)
                        tran_latency = vehicle_i.current_task.data_size / data_rate
                        mig_latency = vehicle_i.current_task.data_size * self.shortest_paths[agent_id][
                            next_agent_id] / next_edge_i.trans_rate  # 传输到当前edge的时延
                        exe_latency = vehicle_i.current_task.CPU_cycle / (next_edge_i.cpu * ratio_cpu)  # 在下一个edge的执行时延
                        wait_latency = 0
                        edge_comp_task_nums[next_agent_id] += 1
                        if next_edge_i.buff_CPU > 0:  # 下一个edge边缘计算是否等待
                            wait_latency = next_edge_i.buff_CPU / next_edge_i.cpu
                        latency = tran_latency + exe_latency + wait_latency + mig_latency
                        # 统计所有需要卸载到下一个edge的任务总量(CPU cycles)
                        all_offload_CPU[next_edge_id] += vehicle_i.current_task.CPU_cycle

                    # 执行车辆的计算: 无论当前任务是否卸载到车辆执行，每个时刻车辆都要执行计算buff上的任务
                    if vehicle_i.buff_CPU < vehicle_i.cpu:
                        vehicle_resource_utilization.append(vehicle_i.buff_CPU / vehicle_i.cpu)
                        vehicle_i.buff_CPU = 0
                    else:
                        vehicle_resource_utilization.append(1)  # 如果执行之后, vehicle的buff不为0, vehicle资源利用率为1
                        vehicle_i.buff_CPU -= vehicle_i.cpu

                    all_latency += latency
                    all_wait_latency += wait_latency
                    all_exe_latency += exe_latency
                    all_tran_latency += tran_latency
                    all_mig_latency += mig_latency

                    # 判断该车任务是否惩罚, 并计算失败率
                    penalty = 0
                    if latency > vehicle_i.current_task.ddl:
                        penalty = latency - vehicle_i.current_task.ddl
                        failure_num += 1

                    agent_rewards.append(latency * (-1) - penalty)

                latencies[agent_id] = all_latency
                wait_latencies[agent_id] = all_wait_latency
                exe_latencies[agent_id] = all_exe_latency
                tran_latencies[agent_id] = all_tran_latency
                mig_latencies[agent_id] = all_mig_latency
                rewards[agent_id] = agent_rewards
                failure_nums[agent_id] = failure_num
                local_nums[agent_id] = local_num

        # 更新每个edge_i.buff_CPU
        for i in all_offload_CPU.keys():
            if all_offload_CPU[i] != 0:
                self.edges[i].buff_CPU += all_offload_CPU[i]

        # 执行每一个edge的计算
        for id_i in self.Edge_IDs:
            edge_i = self.edges[id_i]
            if edge_i.buff_CPU < edge_i.cpu:
                edge_resource_utilization.append(edge_i.buff_CPU / edge_i.cpu)
                edge_i.buff_CPU = 0
            else:
                edge_resource_utilization.append(1)
                edge_i.buff_CPU -= edge_i.cpu

        # action执行完毕，进入下一时刻, 结合模拟数据调整任务
        all_task_num = 0
        # 每个edge覆盖的车辆数--统计每个edge的流量
        edge_task_nums = [0] * self.num_edge

        obs = {}
        share_obs = {}
        next_obs = {}
        next_share_obs = {}
        for id_i, edge_i in self.edges.items():
            if id_i in edge_vehicle_map.keys():
                agent_id = self.Edge_index_map[id_i]
                next_obs[agent_id] = np.zeros([len(edge_vehicle_map[id_i]), self.obs1_dim])
                next_share_obs[agent_id] = np.zeros([len(edge_vehicle_map[id_i]), self.share_obs1_dim])

        whole_edge_state = []
        for id_i, edge_i in self.edges.items():
            whole_edge_state.append([edge_i.cpu, edge_i.buff_CPU, edge_i.bw])
        whole_edge_state = np.array(whole_edge_state).flatten()

        # 对于每一个edge
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            # 车辆根据模拟数据加入vehicle_pool并与Edges连接，更新连接车辆的轨迹预测值
            # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
            edge_i.clear_futuretrajectory()
            if id_i in next_edge_vehicle_map:
                vehicle_ids = next_edge_vehicle_map[id_i]
                for v_id in vehicle_ids:
                    if v_id not in self.vehicle_pool:  # 首次出现的车辆
                        vehicle_i = Vehicle(v_id)  # 新建一个车辆
                        self.vehicle_pool[v_id] = vehicle_i  # 加入系统的车辆pool中
                    if next_future_trajectory[v_id] != None:
                        edge_i.update_futuretrajectory(
                            np.array([self.Edge_index_map[str(int(item)) if isinstance(item, float) else item]
                                      for item in next_future_trajectory[v_id]]))  # 更新预测值
                    else:
                        edge_i.update_futuretrajectory(np.full(self.pred_len, -1))
                all_task_num += len(vehicle_ids)
                edge_task_nums[agent_id] = len(vehicle_ids)  # 统计每个edge的流量
                edge_i.vehicle_connection(vehicle_ids)  # ID 记录至对应edge
            else:
                edge_i.connected_vehicles = []

            # 每个车辆生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # Edge Information
            edge_state = np.array([edge_i.cpu, edge_i.buff_CPU, edge_i.bw])

            if len(edge_i.connected_vehicles) > 0:
                obs[agent_id] = []
                share_obs[agent_id] = []
                # Vehicle information + Task information
                for i, v_id in enumerate(edge_i.connected_vehicles):
                    v_i = self.vehicle_pool[v_id]
                    # [车辆算力，传输功率；任务数据量，任务所需CPU，任务截止时间]
                    vehicle_task_state = [v_i.cpu, v_i.buff_CPU, v_i.trans,
                                          v_i.current_task.data_size, v_i.current_task.CPU_cycle, v_i.current_task.ddl]
                    prediction_state = edge_i.predict_trajectory[i]
                    obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, edge_state)))
                    share_obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, whole_edge_state)))

                    for j in edge_vehicle_map.keys():
                        if v_id in edge_vehicle_map[j]:
                            next_obs[self.Edge_index_map[j]][edge_vehicle_map[j].index(v_id)] = np.hstack(
                                (vehicle_task_state, prediction_state,
                                 np.array([self.edges[j].cpu, self.edges[j].buff_CPU, self.edges[j].bw])))
                            next_share_obs[self.Edge_index_map[j]][edge_vehicle_map[j].index(v_id)] = np.hstack(
                                (vehicle_task_state, prediction_state, whole_edge_state))

                obs[agent_id] = np.array(obs[agent_id])
                share_obs[agent_id] = np.array(share_obs[agent_id])

        info = {}
        info["latencies"] = latencies
        info["wait_latencies"] = wait_latencies
        info["exe_latencies"] = exe_latencies
        info["tran_latencies"] = tran_latencies
        info["mig_latencies"] = mig_latencies
        info["failure_nums"] = failure_nums
        info["local_nums"] = local_nums
        info["average_vehicle_utilization"] = np.mean(vehicle_resource_utilization)
        info["average_edge_utilization"] = np.mean(edge_resource_utilization)

        self.count_step += 1
        if self.count_step >= self.time_slots:
            self.done = True

        return all_task_num, edge_task_nums, edge_comp_task_nums, obs, share_obs, next_obs, next_share_obs, rewards, info, self.done
