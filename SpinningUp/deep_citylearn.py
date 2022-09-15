from SpinningUp.utils.run_utils import ExperimentGrid
import SpinningUp.ddpg.ddpg as ddpg
import SpinningUp.ppo.ppo as ppo
import torch

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ppo-pyt-bench')
    #eg.add('env_name', 'Epoch-Citylearn-v1', '', True)
    eg.add('env_name', 'Cartpole-v1', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.run(ppo, num_cpu=args.cpu)