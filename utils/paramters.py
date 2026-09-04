import argparse
import torch
import os
from utils.gpu_dispatch import GPU
from easydict import EasyDict as edict
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger


def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--epsilon_theta", type=str, default='UGnet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_schedule", type=str, default='quad')  # uniform, quad
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=200)  # sample_steps
    parser.add_argument("--ss", type=str, default='ddpm') #help='sample strategy', ddpm, multi_diffusion, one_diffusion
    parser.add_argument("--T_h", type=int, default=12)

    # eval
    parser.add_argument('--n_samples', type=int, default=8)

    # train
    parser.add_argument("--is_train", type=bool, default=True) # train or evaluate
    parser.add_argument("--data", type=str, default='PEMS08')
    parser.add_argument("--mask_ratio", type=float, default=0.0) # mask of history data
    parser.add_argument("--is_test", type=bool, default=True)
    parser.add_argument("--nni", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--T_p", type=int, default=10)
    parser.add_argument("--V", type=int, default=5)
    parser.add_argument("--points_per_hour", type=int, default=57600)

    args, _ = parser.parse_known_args()
    return args


def get_diffusion_config(data='PEMS08'):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'

    config.data.feature_file = config.data.path + config.data.name + '/flow.npy'
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1
    config.data.points_per_hour = 12
    config.data.num_vertices = 4
    config.data.num_features = 3

    gpu_id = None
    try:
        gpu_id = GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0])
    except Exception:
        pass
    config.gpu_id = gpu_id
    if gpu_id is not None:
        cuda_id = GpuId2CudaId(gpu_id)
        torch.cuda.set_device(f"cuda:{cuda_id}")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_id}" if gpu_id is not None else "cuda:0")
    else:
        device = torch.device("cpu")
    # device = torch.device('cpu')

    # model config
    config.model = edict()

    config.model.T_p = 12
    config.model.T_h = 12
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7
    config.model.day_len = config.data.points_per_hour * 24
    config.model.device = device
    config.model.d_h = 32

    # config for diffusion model
    config.model.N = 200
    config.model.sample_steps = 200
    # config.model.N = 12
    # config.model.sample_steps = 12
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    config.model.beta_end = 0.02
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'

    config.n_samples = 2

    # config for UGnet
    config.model.channel_multipliers = [1, 2]  # The list of channel numbers at each resolution.
    config.model.supports_len = 2

    # training config
    config.model_name = 'DiffSTG'
    config.is_test = False  # Whether run the code in the test mode
    config.epoch =50  # Number of max training epoch
    config.optimizer = "adam"
    config.lr = 1e-4
    config.batch_size = 32
    config.wd = 1e-5
    config.early_stop = 10
    config.start_epoch = 0
    config.device = device
    config.logger = Logger()


    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_FORECAST):
        os.makedirs(config.PATH_FORECAST)

    # Override defaults from command-line arguments (allows run.py or external scripts to override)
    params = vars(get_params())
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.T_h = config.model.T_h = params['T_h']
    config.T_p = config.model.T_p = params['T_h']

    return config
