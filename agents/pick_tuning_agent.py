from agents.day_tunable_agent import MultiPolicyDayAgent
from agents.day_tunable_agent_actions import TunableDayActionsAgent
from agents.tuning_month_agent import MultiPolicyAgent
from agents.tuning_year_agent import TuningYearAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################

# TuningAgent = MultiPolicyAgent
# TuningAgent = MultiPolicyDayAgent
# TuningAgent = TuningYearAgent
TuningAgent = TunableDayActionsAgent