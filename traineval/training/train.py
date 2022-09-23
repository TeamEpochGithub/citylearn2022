import argparse
import os.path as osp

import gym
import torch
from gym.envs.registration import register

from traineval.utils.convert_arguments import environment_convert_argument, get_environment_arguments
from traineval.training.spinningup.ddpg import core as ddpgcore, ddpg
from traineval.training.spinningup.environments import epoch_citylearn
from traineval.training.spinningup.ppo import core as ppocore, ppo
from traineval.training.spinningup.sac import core as saccore, sac
from traineval.training.spinningup.td3 import core as td3core, td3
from traineval.training.spinningup.vpg import core as vpgcore, vpg
from traineval.training.spinningup.utils.mpi_tools import mpi_fork
from traineval.training.spinningup.utils.run_utils import setup_logger_kwargs
from traineval.utils.convert_arguments import get_environment_arguments
from traineval.utils.register_environment import register_environment


class TrainModel:

    def __init__(self, model_args):
        self.model_args = self.retrieve_parsed_args(model_args)
    # TODO: take arguments as input and add them to parser if they are not-None
    def retrieve_parsed_args(self, arguments):
        parser = argparse.ArgumentParser()

        for i in range(len(arguments)):
            parser.add_argument(*arguments[i][0], type=arguments[i][1], default=arguments[i][2])
        # parser.add_argument('--env', type=str, default='Epoch-Citylearn-v1')
        # parser.add_argument('--hid', type=int, default=64)
        # parser.add_argument('--l', type=int, default=2)
        # parser.add_argument('--gamma', type=float, default=0.99)
        # parser.add_argument('--seed', '-s', type=int, default=0)
        # parser.add_argument('--cpu', type=int, default=4)
        # parser.add_argument('--steps', type=int, default=4000)
        #parser.add_argument('--epochs', type=int, default=self.epochs)
        #parser.add_argument('--exp_name', type=str, default=self.model_type)
        # parser.add_argument('--save_freq', type=int, default=1)
        args, unknown = parser.parse_known_args()

        return args

    def run_ppo(self):

        parsed_args = self.model_args

        # CAN'T USE mpi_fork WHEN RUNNING FROM JUPYTER NOTEBOOKS
        mpi_fork(parsed_args.cpu)

        logger_kwargs = setup_logger_kwargs(parsed_args.exp_name, parsed_args.seed)
        ppo.ppo(lambda: gym.make(parsed_args.env), actor_critic=ppocore.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[parsed_args.hid] * parsed_args.l), gamma=parsed_args.gamma,
                seed=parsed_args.seed, steps_per_epoch=parsed_args.steps, epochs=parsed_args.epochs,
                logger_kwargs=logger_kwargs, save_freq=parsed_args.save_freq)

        print("##### PPO model trained #####")

    def run_ddpg(self):

        parsed_args = self.model_args

        logger_kwargs = setup_logger_kwargs(parsed_args.exp_name, parsed_args.seed)
        ddpg.ddpg(lambda: gym.make(parsed_args.env), actor_critic=ddpgcore.MLPActorCritic,
                  ac_kwargs=dict(hidden_sizes=[parsed_args.hid] * parsed_args.l),
                  gamma=parsed_args.gamma, seed=parsed_args.seed, epochs=parsed_args.epochs,
                  logger_kwargs=logger_kwargs, save_freq=parsed_args.save_freq)

        print("##### DDPG model trained #####")

    def run_sac(self):

        parsed_args = self.model_args

        torch.set_num_threads(torch.get_num_threads())

        logger_kwargs = setup_logger_kwargs(parsed_args.exp_name, parsed_args.seed)
        sac.sac(lambda: gym.make(parsed_args.env), actor_critic=saccore.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[parsed_args.hid] * parsed_args.l),
                gamma=parsed_args.gamma, seed=parsed_args.seed, epochs=parsed_args.epochs,
                logger_kwargs=logger_kwargs, save_freq=parsed_args.save_freq)

        print("##### SAC model trained #####")

    def run_td3(self):

        parsed_args = self.model_args

        logger_kwargs = setup_logger_kwargs(parsed_args.exp_name, parsed_args.seed)
        td3.td3(lambda: gym.make(parsed_args.env), actor_critic=td3core.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[parsed_args.hid] * parsed_args.l),
                gamma=parsed_args.gamma, seed=parsed_args.seed, epochs=parsed_args.epochs,
                logger_kwargs=logger_kwargs, save_freq=parsed_args.save_freq)

        print("##### TD3 model trained #####")

    def run_vpg(self):

        parsed_args = self.model_args

        mpi_fork(parsed_args.cpu)

        logger_kwargs = setup_logger_kwargs(parsed_args.exp_name, parsed_args.seed)
        vpg.vpg(lambda: gym.make(parsed_args.env), actor_critic=vpgcore.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[parsed_args.hid] * parsed_args.l), gamma=parsed_args.gamma,
                seed=parsed_args.seed, steps_per_epoch=parsed_args.steps, epochs=parsed_args.epochs,
                logger_kwargs=logger_kwargs, save_freq=parsed_args.save_freq)

        print("##### VPG model trained #####")

    def train_model(self):

        if self.model_args.exp_name == "ppo":
            self.run_ppo()
        elif self.model_args.exp_name == "ddpg":
            self.run_ddpg()
        elif self.model_args.exp_name == "sac":
            self.run_sac()
        elif self.model_args.exp_name == "td3":
            self.run_td3()
        elif self.model_args.exp_name == "vpg":
            self.run_vpg()

    # TODO: Add ExperimentGrid for GridSearchCV-like hyperparameter tuning


if __name__ == "__main__":
    district_args = ["hour",
                     "month",
                     "carbon_intensity",
                     "electricity_pricing",
                     "outdoor_dry_bulb_temperature_predicted_6h",
                     "outdoor_relative_humidity_predicted_6h"]

    building_args = ["non_shiftable_load",
                     "solar_generation",
                     "electrical_storage_soc",
                     "net_electricity_consumption"]

    model_type = "ppo"
    number_of_epochs = 5

    model_args = argument_list = [
        [['--env'], str, 'Epoch-Citylearn-v1'],
        [['--hid'], int, 64],
        [['--l'], int, 2],
        [['--gamma'], float, 0.99],
        [['--seed', '-s'], int, 0],
        [['--cpu'], int, 4],
        [['--steps'], int, 4000],
        [['--epochs'], int, number_of_epochs],
        [['--exp_name'], str, model_type],
        [['--save_freq'], int, 1],
    ]

    environment_arguments = get_environment_arguments(district_args, building_args)

    trainer = TrainModel(model_args)

    register_environment(environment_arguments=environment_arguments)

    trainer.train_model()
