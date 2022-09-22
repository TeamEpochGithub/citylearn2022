import os.path as osp
from gym.envs.registration import register
from traineval.training.spinningup.environments import epoch_citylearn

def register_environment(environment_arguments):
    complete_path = osp.dirname(epoch_citylearn.__file__).replace("\\", ".")
    relative_path = complete_path[complete_path.find("kit.") + 4:] + ".epoch_citylearn"

    register(
        id="Epoch-Citylearn-v1",
        entry_point=relative_path + ":EnvCityGym",
        kwargs=environment_arguments,
    )