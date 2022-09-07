from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent
from agents.sac_agent import SAC
from agents.rlc_agent import RLCAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################

# UserAgent = RandomAgent
# UserAgent = BasicRBCAgent
UserAgent = SAC(RLCAgent)
