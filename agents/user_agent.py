###################################################################
#####                Specify your agent here                  #####
###################################################################
from agents.timestep_known_consumption_agent import TimeStepKnownConsumptionAgent
from agents.timestep_known_consumption_agent_peak import TimeStepKnownConsumptionAgentPeak
from agents.timestep_pred_consumption_agent import TimeStepPredConsumptionAgent
from agents.timestep_pred_consumption_agent_peak import TimeStepPredConsumptionAgentPeak
from agents.timestep_known_consumption_agent_peak_carbon import TimeStepKnownConsumptionAgentPeakCarbon
from agents.timestep_pred_consumption_agent_peak_carbon import TimeStepPredConsumptionAgentPeakCarbon
from agents.timestep_pred_consumption_agent_peak_carbon_error_corr import TimeStepPredConsumptionAgentPeakCarbonErrorCorrection

# UserAgent = RandomAgent
# UserAgent = KnownConsumptionAgent
# UserAgent = ImprovedIndividualConsumptionAgent
# UserAgent = TimeStepKnownConsumptionAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
# UserAgent = ConsumptionBasedAgent
# UserAgent = DayTunedAgent
# UserAgent = TimeStepPredConsumptionAgent
# UserAgent = TimeStepKnownConsumptionAgentPeak
# UserAgent = TimeStepKnownConsumptionAgentPeakCarbon
# UserAgent = TimeStepPredConsumptionAgentPeak
# UserAgent = TimeStepPredConsumptionAgentPeakCarbon
UserAgent = TimeStepPredConsumptionAgentPeakCarbonErrorCorrection

# Changen naar MonthTunedAgent en dan weer submission maken

