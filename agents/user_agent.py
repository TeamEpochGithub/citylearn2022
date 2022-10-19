###################################################################
#####                Specify your agent here                  #####
###################################################################
from agents.known_consumption_agent import KnownConsumptionAgent
from agents.pred_consumption_agent import PredConsumptionAgent
from agents.known_consumption_agent_peak import KnownConsumptionAgentPeak
# from agents.unrefactored_known_consumption_timestep import TimeStepKnownConsumptionAgent

UserAgent = KnownConsumptionAgentPeak
