#from agents.individual_consumption_agent import IndividualConsumptionAgent
from agents.improved_individual_consumption import IndividualConsumptionAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
UserAgent = IndividualConsumptionAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
#UserAgent = ConsumptionBasedAgent
# UserAgent = DayTunedAgent

# Changen naar MonthTunedAgent en dan weer submission maken