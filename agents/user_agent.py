from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent
from agents.basic_reward_agent import BasicRewardAgent
from agents.weather_agent import BasicWeatherAgent
from agents.sac_agent import SAC
from agents.rlc_agent import RLCAgent
from agents.SpinningUpAgent import BasicPPOAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = BasicRBCAgent
# UserAgent = SAC(RLCAgent)
# UserAgent = BasicRewardAgent
# UserAgent = BasicWeatherAgent
UserAgent = BasicPPOAgent