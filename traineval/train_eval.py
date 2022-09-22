from traineval.evaluation.spinning_up_evaluation import evaluate
from traineval.training.train import TrainModel
from traineval.utils.convert_arguments import get_environment_arguments


class TrainerEvaluator:

    def __init__(self, epochs, model_type, model_args):
        self.epochs = epochs
        self.model_type = model_type
        self.model_args = model_args

    def setup_trainer(self, current_environment_arguments):
        current_trainer = TrainModel(self.epochs, self.model_type)
        current_trainer.register_environment(current_environment_arguments)
        return current_trainer

    def run_trainer(self, trainer):
        # TODO: run_ppo should take arguments
        trainer.train_model(trainer, self.model_args)

    def run_evaluation(self, environment_arguments, model_type, model_seed, model_iteration):
        return evaluate(environment_arguments, model_type, model_seed, model_iteration)


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

    model_args = argument_list = [
        [['--env'], str, 'Epoch-Citylearn-v1'],
        [['--hid'], int, 64],
        [['--l'], int, 2],
        [['--gamma'], float, 0.99],
        [['--seed', '-s'], int, 0],
        [['--cpu'], int, 4],
        [['--steps'], int, 4000],
        [['--save_freq'], int, 1],
        [['--num_runs'], int, 1]
        ]

    environment_arguments = get_environment_arguments(district_args, building_args)

    model_type = "ppo"
    num_epochs = 10
    trainer_evaluator = TrainerEvaluator(epochs=num_epochs, model_type=model_type, model_args=model_args)
    trainer = trainer_evaluator.setup_trainer(current_environment_arguments=environment_arguments)
    trainer_evaluator.run_trainer(trainer)

    averaged_score, agent_time_elapsed = trainer_evaluator.run_evaluation(environment_arguments=environment_arguments,
                                                                          model_type=model_type, model_seed="0",
                                                                          model_iteration=str(num_epochs - 1))
