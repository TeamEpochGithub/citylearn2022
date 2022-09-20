import argparse

import gym
import os.path as osp

from gym.envs.registration import register

from train_eval.utils.convert_arguments import environment_convert_argument
from train_eval.training.spinningup.ddpg import ddpg
from train_eval.training.spinningup.ddpg import core as ddpgcore
from train_eval.training.spinningup.environments import epoch_citylearn
from train_eval.training.spinningup.ppo import core as ppocore, ppo
from train_eval.training.spinningup.utils.mpi_tools import mpi_fork
from train_eval.training.spinningup.utils.run_utils import setup_logger_kwargs


class TrainModel:

    def __init__(self, epochs):
        self.epochs = epochs

    def register_environment(self, environment_arguments):
        complete_path = osp.dirname(epoch_citylearn.__file__).replace("\\", ".")
        relative_path = complete_path[complete_path.find("kit.") + 4:] + ".epoch_citylearn"

        register(
            id="Epoch-Citylearn-v1",
            entry_point=relative_path + ":EnvCityGym",
            kwargs=environment_arguments,
        )

    def retrieve_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='Epoch-Citylearn-v1')
        parser.add_argument('--hid', type=int, default=64)
        parser.add_argument('--l', type=int, default=2)
        parser.parse_args()
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--seed', '-s', type=int, default=0)
        parser.add_argument('--cpu', type=int, default=4)
        parser.add_argument('--steps', type=int, default=4000)
        parser.add_argument('--epochs', type=int, default=self.epochs)
        parser.add_argument('--exp_name', type=str, default='ppo')
        args = parser.parse_args()

        return args

    def run_ppo(self):
        args = self.retrieve_args()

        mpi_fork(args.cpu)

        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
        ppo.ppo(lambda: gym.make(args.env), actor_critic=ppocore.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
                seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
                logger_kwargs=logger_kwargs)

        print("##### PPO model trained #####")

    def run_ddpg(self):

        args = self.retrieve_args()

        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
        ddpg.ddpg(lambda: gym.make(args.env), actor_critic=ddpgcore.MLPActorCritic,
                  ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                  gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                  logger_kwargs=logger_kwargs)

        print("##### DDPG model trained #####")

    # def run_experiment_grid(self):
    #     args = self.retrieve_args()
    #
    #     eg = ExperimentGrid(name='ppo-pyt-bench')
    #     eg.add('env_name', 'Epoch-Citylearn-v1', '', True)
    #     eg.add('seed', [10 * i for i in range(args.num_runs)])
    #     eg.add('epochs', 10)
    #     eg.add('steps_per_epoch', 4000)
    #     eg.add('ac_kwargs:hidden_sizes', [(32,), (64, 64)], 'hid')
    #     eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    #     eg.run(ppo_pytorch, num_cpu=args.cpu)
    #
    #     print("##### Experiment grid ran #####")



if __name__ == "__main__":
    trainer = TrainModel(epochs=100)

    environment_arguments = {
        "district_indexes": [0, 2, 19, 4, 8, 24],
        "district_normalizers": [12, 24, 2, 100, 100, 1],
        "building_indexes": [20, 21, 22, 23],
        "building_normalizers": [5, 5, 5, 5]}


    print(environment_convert_argument([0,2,19]))
    print(environment_convert_argument(["month", "hour", "solar_generation"]))

    trainer.register_environment(environment_arguments=environment_arguments)
    trainer.run_ppo()
    # trainer.run_ddpg()
    # trainer.run_experiment_grid()
