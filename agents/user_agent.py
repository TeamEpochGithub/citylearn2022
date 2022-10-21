###################################################################
#####                Specify your agent here                  #####
###################################################################
from agents.timestep_known_consumption_agent import TimeStepKnownConsumptionAgent
from agents.timestep_known_consumption_agent_peak import TimeStepKnownConsumptionAgentPeak
from agents.timestep_pred_consumption_agent import TimeStepPredConsumptionAgent
from agents.timestep_pred_consumption_agent_peak import TimeStepPredConsumptionAgentPeak

# UserAgent = RandomAgent
# UserAgent = KnownConsumptionAgent
# UserAgent = ImprovedIndividualConsumptionAgent
# UserAgent = TimeStepKnownConsumptionAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
#UserAgent = ConsumptionBasedAgent
# UserAgent = DayTunedAgent
UserAgent = TimeStepPredConsumptionAgent
# UserAgent = TimeStepKnownConsumptionAgentPeak

# Changen naar MonthTunedAgent en dan weer submission maken

