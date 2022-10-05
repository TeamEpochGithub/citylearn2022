from agents.tuning_month_agent import MultiPolicyAgent
from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent
from agents.basic_reward_agent import BasicRewardAgent
from agents.weather_agent import BasicWeatherAgent
from agents.sac_agent import SAC
from agents.rlc_agent import RLCAgent
from agents.spinning_up_agent import BasicPPOAgent
from agents.month_tuned_agent import MonthTunedAgent
from agents.day_tuned_agent import DayTunedAgent
from agents.tuning_month_agent import MultiPolicyAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = BasicRBCAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
UserAgent = MonthTunedAgent
# UserAgent = DayTunedAgent

# Changen naar MonthTunedAgent en dan weer submission maken