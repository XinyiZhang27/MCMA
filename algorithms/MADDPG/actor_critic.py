import torch
import torch.nn as nn
from algorithms.util import init, check
from myutils.util import get_dim_from_space
from algorithms.MADDPG.utils.mlp import MLPBase
from algorithms.MADDPG.utils.act import ACTLayer


class MADDPG_Actor(nn.Module):
    """
    Actor network class for MADDPG/MATD3. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_dim, act_dim, device):
        super(MADDPG_Actor, self).__init__()

        self.hidden_size = args.hidden_size2
        self._gain = args.gain2
        self._use_orthogonal = args.use_orthogonal2
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.mlp = MLPBase(args, obs_dim)
        self.act = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, self._gain)
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def forward(self, x):
        """
        Compute actions using the needed information.
        :param x: (np.ndarray) Observations with which to compute actions.
        """
        x = check(x).to(**self.tpdv)
        x = self.mlp(x)
        x = self.act(x)
        # action = self.tanh(x)
        action = self.sigmoid(x)

        return action


class MADDPG_Critic(nn.Module):
    """
    Critic network class for MADDPG/MATD3. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param share_obs_dim: (int) dimension of the centralized observation vector.
    :param share_act_dim: (int) dimension of the centralized action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param num_q_outs: (int) number of q values to output (1 for MADDPG, 2 for MATD3).
    """
    def __init__(self, args, share_obs_dim, share_act_dim, device):
        super(MADDPG_Critic, self).__init__()
        self.hidden_size = args.hidden_size2
        self._use_orthogonal = args.use_orthogonal2
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        input_dim = share_obs_dim + share_act_dim
        self.mlp = MLPBase(args, input_dim)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_outs = init_(nn.Linear(self.hidden_size, 1)).to(device)
        
        self.to(device)

    def forward(self, central_obs, central_act):
        """
        Compute Q-values using the needed information.
        :param central_obs: (np.ndarray) Centralized observations with which to compute Q-values.
        :param central_act: (np.ndarray) Centralized actions with which to compute Q-values.

        :return q_values: (list) Q-values outputted by each Q-network.
        """
        central_obs = check(central_obs).to(**self.tpdv)
        central_act = check(central_act).to(**self.tpdv)

        x = torch.cat([central_obs, central_act], dim=1)

        x = self.mlp(x)
        q_values = self.q_outs(x)

        return q_values
