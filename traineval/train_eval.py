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

    def run_trainer(self, trainer):
        # TODO: run_ppo should take arguments
        trainer.run_ppo()

    def run_evaluation(self):
        evaluate()


if __name__ == "__main__":

    district_args = ["hour",
                     "month",
                     "carbon_intensity",
                     "electricity_pricing"]

    building_args = ["non_shiftable_load",
                     "solar_generation",
                     "electrical_storage_soc",
                     "net_electricity_consumption"]

    environment_arguments = get_environment_arguments(district_args, building_args)

    trainer_evaluator = TrainerEvaluator(200)
    trainer = trainer_evaluator.setup_trainer(environment_arguments=environment_arguments)
    trainer_evaluator.run_trainer(trainer)

    # TODO: fix it so you can evaluate from here (probably best to turn the model output file into package so you can import it)
    # trainer_evaluator.run_evaluation(environment_arguments=environment_arguments, agent_file_name="model0.pt")

    # Trainer wrapper should return model and time taken to achieve model every time it saves
    # Then we run evaluation on model
    # Finally we return all times_taken and average_scores to plot them
