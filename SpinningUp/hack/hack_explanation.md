How to train models with spinning up.

1: Copy the epoch_citylearn.py file into `venv/lib/site-packages/gym/envs/classic_control`

2: Edit the `venv/lib/site-packages/gym/envs/classic_control/__init__.py` file to contain the following:

    register(
        id="Epoch-Citylearn-v0",
        entry_point="gym.envs.classic_control.epoch_citylearn:EnvCityGym",
    )