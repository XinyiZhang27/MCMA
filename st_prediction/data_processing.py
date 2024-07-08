import numpy as np
import pandas as pd
import os
import pickle
import random
from datetime import datetime
from myutils.timefeatures import time_features


def check_and_create_path(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


#  每个时刻有哪些车在系统内
def generate_time_vehicle_relationship(data_file, save_file):
    data = pd.read_csv(data_file, dtype={"time": int, "vehicle_id": str, "pos_x": float, "pos_y": float})
    time_grouped_data = data.groupby("time")
    time_vehicle = {}
    for time, current_data in time_grouped_data:
        time_vehicle[time] = current_data['vehicle_id'].unique().tolist()
    check_and_create_path(save_file)
    fw = open(save_file, "wb")
    pickle.dump(time_vehicle, fw)


# 记录每一辆车的轨迹 并 划分序列(有重叠序列)
def generate_trajectory_data(data_file, save_file_1, save_file_2, seq_len, label_len, pred_len):
    data = pd.read_csv(data_file, dtype={"time": int, "vehicle_id": str, "pos_x": float, "pos_y": float})
    # 将time转换为时间戳
    current_date = datetime.now().date()
    current_datetime = datetime.combine(current_date, datetime.min.time())  # 将日期转换为 datetime 对象
    data["time"] = current_datetime + pd.to_timedelta(data["time"], unit='s')

    vehicle_grouped_data = data.groupby("vehicle_id")
    vehicle_num = len(vehicle_grouped_data)
    print("vehicle num: ", vehicle_num)

    vehicle_trajectory = {}  # 记录每辆车的start_time, time features序列和trajectory(pos_x, pos_y)序列
    seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list = [], [], [], []
    counter = 0
    # 对每一辆车的轨迹进行划分
    for vehicle, current_data in vehicle_grouped_data:
        current_data = current_data.sort_values(by="time", ascending=True)
        trajectory = current_data.iloc[:, -2:].values
        times = current_data[["time"]]
        t_features = time_features(times, freq='s')

        time_and_trajectory = {}
        time_and_trajectory["start_time"] = int((current_data["time"].min() - current_datetime).total_seconds())
        time_and_trajectory["time_features"] = t_features
        time_and_trajectory["trajectory"] = trajectory
        vehicle_trajectory[vehicle] = time_and_trajectory

        time_len = t_features.shape[0]
        for time_i in range(0, time_len-seq_len-pred_len+1):
            x_begin = time_i
            x_end = x_begin + seq_len
            y_begin = x_end - label_len
            y_end = y_begin + label_len + pred_len

            seq_x_list.append(trajectory[x_begin:x_end])
            seq_x_mark_list.append(t_features[x_begin:x_end])
            seq_y_list.append(trajectory[y_begin:y_end])
            seq_y_mark_list.append(t_features[y_begin:y_end])

    # 对数据进行打乱
    random.shuffle(seq_x_list)
    random.shuffle(seq_x_mark_list)
    random.shuffle(seq_y_list)
    random.shuffle(seq_y_mark_list)
    all_num = len(seq_x_list)
    print(len(seq_x_list))
    print(len(seq_x_mark_list))
    print(len(seq_y_list))
    print(len(seq_y_mark_list))

    # 切分训练 验证 测试集  {'train': 0, 'val': 1, 'test': 2} --- 70% 训练, 10%验证, 20&测试
    type_map = {0: 'train', 1: 'val', 2: 'test'}
    num_train = int(all_num * 0.7)
    num_vali = int(all_num * 0.1)
    border1s = [0, num_train, num_train + num_vali]
    border2s = [num_train, num_train + num_vali, all_num]
    for i in range(0, 3):
        border1 = border1s[i]
        border2 = border2s[i]
        seqs_x = seq_x_list[border1:border2]
        seqs_x_mark = seq_x_mark_list[border1:border2]
        seqs_y = seq_y_list[border1:border2]
        seqs_y_mark = seq_y_mark_list[border1:border2]
        save_file_3 = os.path.join(save_file_2, type_map[i]+"_trajectory_sequence_seq{}label{}pre{}.npz".format(
                                   seq_len, label_len, pred_len))
        check_and_create_path(save_file_3)
        np.savez(save_file_3, seqs_x=np.array(seqs_x), seqs_x_mark=np.array(seqs_x_mark),
                 seqs_y=np.array(seqs_y), seqs_y_mark=np.array(seqs_y_mark))

    check_and_create_path(save_file_1)
    fw = open(save_file_1, "wb")
    pickle.dump(vehicle_trajectory, fw)


if __name__ == "__main__":
    # Scenario： 3-3 grid
    Simulation_scenario = "3-3-grid"
    Time_range = "10h"

    # Scenario： Net4
    # Simulation_scenario = "Net4"
    # Time_range = "24h"

    # Scenario： bologna_pasubio
    # Simulation_scenario = "bologna_pasubio"
    # Time_range = "24h"

    # Scenario： bologna_acosta
    # Simulation_scenario = "bologna_acosta"
    # Time_range = "24h"

    # 记录每个时刻有哪些车在系统内time_vehicle.pkl --- key: start_time, value: list of vehicle_id
    save_data_file = "trajectory_data/{}/{}/time_vehicle.pkl".format(Simulation_scenario, Time_range)
    load_data_file = "../sumo/data/{}/{}/vehicle_simulation.csv".format(Simulation_scenario, Time_range)
    generate_time_vehicle_relationship(load_data_file, save_data_file)

    # 记录每一辆车的轨迹 --- key: vehicle_id, value: time_and_trajectory --- keys: start_time, time_features和trajectory
    save_data_file1 = "trajectory_data/{}/{}/vehicle_trajectory.pkl".format(Simulation_scenario, Time_range)
    # 划分序列(有重叠序列)
    seq_len = 32
    label_len = 16
    pred_len = 8
    save_data_file2 = "trajectory_data/{}/{}".format(Simulation_scenario, Time_range)
    load_data_file = "../sumo/data/{}/{}/vehicle_simulation.csv".format(Simulation_scenario, Time_range)
    generate_trajectory_data(load_data_file, save_data_file1, save_data_file2, seq_len, label_len, pred_len)
