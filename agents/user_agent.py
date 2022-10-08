from agents.combined_policy_agent import MultiPolicyAgent
from agents.consumption_based_agent import ConsumptionBasedAgent
from agents.month_tuned_agent import MonthTunedAgent
from agents.new_agents.rbc_agent import BasicRBCAgent, BasicRBCAgent2


###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
UserAgent = BasicRBCAgent2
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
#UserAgent = ConsumptionBasedAgent
# UserAgent = DayTunedAgent

# Changen naar MonthTunedAgent en dan weer submission maken