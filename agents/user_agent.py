#from agents.individual_consumption_agent import IndividualConsumptionAgent
from agents.improved_individual_consumption import ImprovedIndividualConsumptionAgent
from agents.known_consumption_agent import KnownConsumptionAgent
from agents.improved_individual_consumption import ImprovedIndividualConsumptionAgent
from agents.timestep_known_consumption_agent import TimeStepKnownConsumptionAgent
from agents.timestep_pred_consumption_agent import TimeStepPredConsumptionAgent
from agents.known_consumption_agent import KnownConsumptionAgent
from agents.pred_consumption_agent import PredConsumptionAgent
from agents.timestep_known_consumption_agent_peak import TimeStepKnownConsumptionAgentPeak
from agents.timestep_known_consumption_agent_peak_carbon import TimeStepKnownConsumptionAgentPeakCarbon
###################################################################
#####                Specify your agent here                  #####
###################################################################

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