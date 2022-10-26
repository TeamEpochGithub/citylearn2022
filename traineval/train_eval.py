from traineval.training.train import TrainModel
from traineval.utils.convert_arguments import get_environment_arguments
from traineval.utils.register_environment import register_environment


class TrainerEvaluator:

    def __init__(self, model_args, environment_arguments):
        self.environment_arguments = environment_arguments
        self.model_args = model_args

    def setup_trainer(self):
        current_trainer = TrainModel(self.model_args)
        register_environment(self.environment_arguments)
        return current_trainer

    def run_trainer(self, trainer):
        # TODO: run_ppo should take arguments
        trainer.train_model()

    def run_evaluation(self, model_type, model_seed, model_iteration, verbose=True):
        return evaluate(self.environment_arguments, model_type, model_seed, model_iteration, verbose)


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
    number_of_epochs = 50
    model_seed = 0
    save_freq = 20
    steps = 8760
    max_ep_len = steps - 1

    model_args = [
        [['--env'], str, 'Epoch-Citylearn-v1'],
        [['--hid'], int, 64],
        [['--l'], int, 2],
        [['--gamma'], float, 0.99],
        [['--seed', '-s'], int, model_seed],
        [['--cpu'], int, 1],
        [['--steps'], int, steps],
        [['--epochs'], int, number_of_epochs],
        [['--exp_name'], str, model_type],
        [['--max_ep_len'], int, max_ep_len],
        [['--save_freq'], int, save_freq],
        ]

    environment_arguments = get_environment_arguments(district_args, building_args)

    trainer_evaluator = TrainerEvaluator(model_args, environment_arguments)
    trainer = trainer_evaluator.setup_trainer()
    trainer_evaluator.run_trainer(trainer)
    #
    averaged_score, agent_time_elapsed = trainer_evaluator.run_evaluation(model_type=model_type, model_seed=model_seed,
                                                                          model_iteration=20,
                                                                          verbose=True)
    print(averaged_score, agent_time_elapsed)
