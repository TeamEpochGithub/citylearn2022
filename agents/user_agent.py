from agents.consumption_based_tpot_actions_agent import ConsumptionBasedTPOTActionsAgent
from agents.improved_individual_consumption import ImprovedIndividualConsumptionAgent
from agents.improved_individual_consumption_live_learning import ImprovedIndividualConsumptionLiveLearningAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################
from agents.live_learning_consumption_based_agent import LiveLearningAgentBuilder

UserAgent = ImprovedIndividualConsumptionLiveLearningAgent

# UserAgent = ConsumptionBasedTPOTActionsAgent