import argparse


def get_config():

    parser = argparse.ArgumentParser(
        description="mappo_maddpg", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # #Common parameters
    # prepare parameters
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    """
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    """
    # env parameter
    """
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )
    """
    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_true",
        default=False,
        help="Whether agent share the same policy",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="time duration between continuous twice models saving.",
    )
    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="time duration between continuous twice log printing.",
    )

    # #MAPPO
    # replay buffer parameters
    parser.add_argument("--buffer_length1", type=int, default=30000)
    # network parameters
    parser.add_argument(
        "--use_centralized_Q1",
        action="store_false",
        default=True,
        help="Whether to use centralized Q function",
    )
    parser.add_argument(
        "--stacked_frames1",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--hidden_size1",
        type=int,
        default=128,
        help="Dimension of hidden layers for actor/critic networks (default: 64)",
    )
    parser.add_argument(
        "--layer_N1",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_ReLU1",
        action="store_false",
        default=True,
        help="Whether to use ReLU")
    parser.add_argument(
        "--use_feature_normalization1",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal1",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--gain1", type=float, default=0.01, help="The gain # of last action layer")
    parser.add_argument(
        "--use_popart1",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--recurrent_N",
        type=int,
        default=1,
        help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )
    # optimizer parameters
    parser.add_argument(
        "--lr1", type=float, default=15e-5, help="learning rate (default: 5e-4)")
    parser.add_argument(
        "--opti_eps1",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay1", type=float, default=0)
    # ppo parameters
    parser.add_argument(
        "--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=5,
        help="number of batches for ppo (default: 5)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm1",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm1",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma1",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss1",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument(
        "--huber_delta1", type=float, default=10.0, help=" coefficience of huber loss.")
    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_false",
        default=True,
        help="use a linear schedule on the learning rate",
    )
    # pretrained parameters
    parser.add_argument(
        "--model_dir1",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    # #MADDPG
    # replay buffer parameters
    parser.add_argument(
        '--buffer_length2', type=int, default=30000, help="Max # of transitions that replay buffer can contain")
    # network parameters
    parser.add_argument(
        "--use_centralized_Q2",
        action='store_false',
        default=True,
        help="Whether to use centralized Q function")
    parser.add_argument(
        "--stacked_frames2",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument(
        '--hidden_size2',
        type=int,
        default=128,
        help="Dimension of hidden layers for actor/critic networks (default=64)")
    parser.add_argument(
        '--layer_N2',
        type=int,
        default=1,
        help="Number of layers for actor/critic networks")
    parser.add_argument(
        '--use_ReLU2',
        action='store_false',
        default=True,
        help="Whether to use ReLU")
    parser.add_argument(
        '--use_feature_normalization2',
        action='store_false',
        default=True,
        help="Whether to apply layernorm to the inputs")
    parser.add_argument(
        '--use_orthogonal2',
        action='store_false',
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument(
        "--gain2", type=float, default=0.01, help="The gain # of last action layer")
    parser.add_argument(
        "--use_conv1d",
        action='store_true',
        default=False,
        help="Whether to use conv1d")
    parser.add_argument(
        '--use_reward_normalization',
        action='store_false',
        default=True,
        help="Whether to normalize rewards in replay buffer (default=False)")
    parser.add_argument(
        '--use_popart2',
        action='store_true',
        default=False,
        help="Whether to use popart to normalize the target loss")
    parser.add_argument(
        '--popart_update_interval_step',
        type=int,
        default=2,
        help="After how many train steps popart should be updated")
    # optimizer parameters
    parser.add_argument(
        '--lr2_actor', type=float, default=1e-5, help="Learning rate for Adam (default=5e-4)")
    parser.add_argument(
        '--lr2_critic', type=float, default=5e-5, help="Learning rate for Adam (default=5e-4)")
    parser.add_argument(
        "--opti_eps2",
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay2", type=float, default=0)
    # ddpg parameters
    parser.add_argument(
        '--batch_size', type=int, default=256, help="Number of buffer transitions to train on at once (default=32)")
    parser.add_argument(
        "--use_max_grad_norm2", action='store_false', default=True)
    parser.add_argument(
        "--max_grad_norm2", type=float, default=10.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--gamma2', type=float, default=0.99, help="Discount factor for env")
    parser.add_argument(
        '--use_huber_loss2', action='store_true', default=False, help="Whether to use Huber loss for critic update")
    parser.add_argument("--huber_delta2", type=float, default=10.0)
    # soft update parameters
    parser.add_argument('--use_soft_update', action='store_false',
                        default=True, help="Whether to use soft update")
    parser.add_argument('--tau', type=float, default=0.001,
                        help="Polyak update rate (default=0.005)")
    # hard update parameters
    parser.add_argument('--hard_update_interval', type=int, default=200,
                        help="After how many episodes the lagging target should be updated")
    # exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help="Starting value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_finish', type=float, default=0.05,
                        help="Ending value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_anneal_time', type=int, default=50000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument('--act_noise_std', type=float, default=0.2, help="Action noise")
    # pretained parameters
    parser.add_argument("--model_dir2", type=str, default=None)

    return parser
