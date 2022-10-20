###################################################################
#####                Specify your agent here                  #####
###################################################################
from agents.timestep_known_consumption_agent import TimeStepKnownConsumptionAgent
from agents.timestep_known_consumption_agent_peak import TimeStepKnownConsumptionAgentPeak
from agents.timestep_pred_consumption_agent import TimeStepPredConsumptionAgent
from agents.timestep_pred_consumption_agent_peak import TimeStepPredConsumptionAgentPeak

UserAgent = TimeStepPredConsumptionAgent
