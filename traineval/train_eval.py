from traineval.training.train import TrainModel
from traineval.evaluation.spinning_up_evaluation import evaluate
from agents import rbc_agent
from traineval.utils.convert_arguments import environment_convert_argument, environment_convert_scalars, \
    get_environment_arguments


class TrainerEvaluator:

    def __init__(self, epochs):
        self.epochs = epochs

    def setup_trainer(self, environment_arguments):
        trainer = TrainModel(self.epochs)
        trainer.register_environment(environment_arguments)
        return trainer

    def run_trainer(self, trainer, model_type):
        # TODO: run_ppo should take arguments
        trainer.train_model(trainer, model_type=model_type)

    def run_evaluation(self, environment_arguments, model_type="ppo", model_seed="0", model_iteration="3"):
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

    trainer_evaluator = TrainerEvaluator(epochs=10)
    trainer = trainer_evaluator.setup_trainer(environment_arguments=environment_arguments)
    trainer_evaluator.run_trainer(trainer, model_type="td3")

    averaged_score = trainer_evaluator.run_evaluation(environment_arguments=environment_arguments,
                                                      model_type="ppo", model_seed="0", model_iteration="99")
    print(averaged_score)
    # Trainer wrapper should return model and time taken to achieve model every time it saves
    # Then we run evaluation on model
    # Finally we return all times_taken and average_scores to plot them
