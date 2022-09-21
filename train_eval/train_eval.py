from train_eval.training.train import TrainModel
from train_eval.evaluation.local_evaluation_spinning_up import evaluate
from agents import rbc_agent
from train_eval.utils.convert_arguments import environment_convert_argument


class TrainerEvaluator:

    def __init__(self, epochs):
        self.epochs = epochs

    def setup_trainer(self, environment_arguments):
        trainer = TrainModel(self.epochs)
        trainer.register_environment(environment_arguments)
        return trainer

    def run_trainer(self, trainer):

        # TODO: run_ppo should take arguments
        trainer.run_ppo()


    # def setup_evaluator(self):
    #     evaluate()


if __name__ == "__main__":

    district_args = environment_convert_argument(["hour",
                                                  "month",
                                                  "day",
                                                  "carbon_intensity"])
    building_args = environment_convert_argument(["non_shiftable_load",
                                                  "solar_generation",
                                                  "electrical_storage_soc",
                                                  "net_electricity_consumption"])
    environment_arguments = {
        "district_indexes": district_args,
        "district_normalizers": [12, 24, 2, 100, 100, 1],
        "building_indexes": building_args,
        "building_normalizers": [5, 5, 5, 5]}

    trainer_evaluator = TrainerEvaluator(200)
    trainer = trainer_evaluator.setup_trainer(environment_arguments=environment_arguments)
    trainer_evaluator.run_trainer(trainer)

