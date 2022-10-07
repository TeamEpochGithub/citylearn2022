from agents.combined_policy_agent import MultiPolicyAgent
from agents.consumption_based_agent import ConsumptionBasedAgent
from agents.month_tuned_agent import MonthTunedAgent
from agents.month_tuned_agent_nonranges import MonthTunedAgentNoRanges

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = BasicRBCAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
# UserAgent = MonthTunedAgent
UserAgent = MonthTunedAgentNoRanges
# UserAgent = DayTunedAgent

# Changen naar MonthTunedAgent en dan weer submission maken