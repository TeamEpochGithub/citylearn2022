###################################################################
#####                Specify your agent here                  #####
###################################################################
from agents.known_consumption_agent import KnownConsumptionAgent
from agents.pred_consumption_agent import PredConsumptionAgent
from agents.deprecated_agents.live_learning_consumption_agent import LiveLearningConsumptionAgent
from agents.timestep_pred_consumption_agent import TimeStepPredConsumptionAgent

UserAgent = TimeStepPredConsumptionAgent
