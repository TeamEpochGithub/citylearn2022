from agents.combined_policy_agent import MultiPolicyAgent
from agents.consumption_based_agent import ConsumptionBasedAgent
from agents.consumption_pred_agent import ConsumptionPredAgent
from agents.live_learning_consumption_based_agent import LiveLearningAgent, LiveLearningAgentBuilder
from agents.month_tuned_agent import MonthTunedAgent
from agents.month_tuned_agent_nonranges import MonthTunedAgentNoRanges
from agents.consumption_pred_specific_agent import ConsumptionPredAgent2

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = BasicRBCAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
# UserAgent = MonthTunedAgent
# UserAgent = MonthTunedAgentNoRanges
# UserAgent = ConsumptionBasedAgent
UserAgent = LiveLearningAgentBuilder