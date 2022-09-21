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

    # TODO: take arguments as input and add them to parser if they are not-None
    def retrieve_parsed_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='Epoch-Citylearn-v1')
        parser.add_argument('--hid', type=int, default=64)
        parser.add_argument('--l', type=int, default=2)
        parser.parse_args()
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--seed', '-s', type=int, default=0)
        parser.add_argument('--cpu', type=int, default=4)
        parser.add_argument('--steps', type=int, default=4000)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--exp_name', type=str, default='ppo')
        args = parser.parse_args()

        return args

    def run_ppo(self):
        parsed_args = self.retrieve_parsed_args()

        mpi_fork(parsed_args.cpu)

        logger_kwargs = setup_logger_kwargs(parsed_args.exp_name, parsed_args.seed)
        ppo.ppo(lambda: gym.make(parsed_args.env), actor_critic=ppocore.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[parsed_args.hid] * parsed_args.l), gamma=parsed_args.gamma,
                seed=parsed_args.seed, steps_per_epoch=parsed_args.steps, epochs=parsed_args.epochs,
                logger_kwargs=logger_kwargs)

        print("##### PPO model trained #####")

    def run_ddpg(self):

        parsed_args = self.retrieve_parsed_args()

        logger_kwargs = setup_logger_kwargs(parsed_args.exp_name, parsed_args.seed)
        ddpg.ddpg(lambda: gym.make(parsed_args.env), actor_critic=ddpgcore.MLPActorCritic,
                  ac_kwargs=dict(hidden_sizes=[parsed_args.hid] * parsed_args.l),
                  gamma=parsed_args.gamma, seed=parsed_args.seed, epochs=parsed_args.epochs,
                  logger_kwargs=logger_kwargs)

        print("##### DDPG model trained #####")

    def train_model(self, model_type, environment_arguments):

        self.register_environment(environment_arguments=environment_arguments)

        if model_type == "ppo":
            trainer.run_ppo()
        elif model_type == "ddpg":
            trainer.run_ddpg()

    # TODO: Add ExperimentGrid for GridSearchCV-like hyperparameter tuning


if __name__ == "__main__":
    trainer = TrainModel(epochs=100)

    environment_arguments = {
        "district_indexes": [0, 2, 19, 4, 8, 24],
        "district_normalizers": [12, 24, 2, 100, 100, 1],
        "building_indexes": [20, 21, 22, 23],
        "building_normalizers": [5, 5, 5, 5]}

    trainer.register_environment(environment_arguments=environment_arguments)
    trainer.run_ppo()
    # trainer.run_ddpg()
    # trainer.run_experiment_grid()
