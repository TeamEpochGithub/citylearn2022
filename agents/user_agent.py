###################################################################
#####                Specify your agent here                  #####
###################################################################
from agents.timestep_known_consumption_agent_peak import TimeStepKnownConsumptionAgentPeak
from agents.timestep_pred_consumption_agent_peak import TimeStepPredConsumptionAgentPeak
from agents.vowpal_wabbit.contextual_bandit import ContextualBanditAgent

UserAgent = ContextualBanditAgent

