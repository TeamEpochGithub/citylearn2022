#from agents.individual_consumption_agent import IndividualConsumptionAgent
from agents.improved_individual_consumption import ImprovedIndividualConsumptionAgent
from agents.known_consumption_agent import KnownConsumptionAgent
from agents.improved_individual_consumption import ImprovedIndividualConsumptionAgent
from agents.timestep_known_consumption_agent import TimeStepKnownConsumptionAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = KnownConsumptionAgent
# UserAgent = ImprovedIndividualConsumptionAgent
UserAgent = TimeStepKnownConsumptionAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
#UserAgent = ConsumptionBasedAgent
# UserAgent = DayTunedAgent

# Changen naar MonthTunedAgent en dan weer submission maken