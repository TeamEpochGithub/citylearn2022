from traineval.evaluation.spinning_up_evaluation import evaluate
from traineval.training.train import TrainModel
from traineval.utils.convert_arguments import get_environment_arguments


class TrainerEvaluator:

    def __init__(self, epochs):
        self.epochs = epochs

    def setup_trainer(self, current_environment_arguments):
        current_trainer = TrainModel(self.epochs)
        current_trainer.register_environment(current_environment_arguments)
        return current_trainer

    def run_trainer(self, trainer, model_type):
        # TODO: run_ppo should take arguments
        trainer.train_model(trainer, model_type=model_type)

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

    environment_arguments = get_environment_arguments(district_args, building_args)

    model_type = "ppo"
    num_epochs = 1000
    trainer_evaluator = TrainerEvaluator(epochs=num_epochs)
    trainer = trainer_evaluator.setup_trainer(current_environment_arguments=environment_arguments)
    trainer_evaluator.run_trainer(trainer, model_type=model_type)

    averaged_score, agent_time_elapsed = trainer_evaluator.run_evaluation(environment_arguments=environment_arguments,
                                                                          model_type=model_type, model_seed="0",
                                                                          model_iteration=str(num_epochs - 1))
