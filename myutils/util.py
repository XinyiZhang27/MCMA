import numpy as np
import math
import os
import pandas as pd
import pickle
import torch
from gym.spaces import Box, Discrete, Tuple


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2/2


def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, Discrete):
        dim = space.n
    elif isinstance(space, Tuple):
        dim = sum([get_dim_from_space(sp) for sp in space])
    elif "MultiDiscrete" in space.__class__.__name__:
        return (space.high - space.low) + 1
    elif isinstance(space, list):
        dim = space[0]
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def check_and_create_path(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


def load_edge_ids(edge_data_file):
    edge_data = pd.read_csv(edge_data_file,dtype={"edge_id": str,"edge_pos_x":float, "edge_pos_y":float})
    return edge_data["edge_id"].tolist()


def load_pkl(data_file):
    fr = open(data_file, "rb")
    result = pickle.load(fr)
    return result

def save_pkl(save_path, data):
    check_and_create_path(save_path)
    fw = open(save_path, "wb")
    pickle.dump(np.array(data), fw)
