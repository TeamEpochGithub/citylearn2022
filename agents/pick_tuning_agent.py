from agents.tunable_agent import MultiPolicyAgent
from agents.day_tunable_agent import MultiPolicyDayAgent
from agents.tuning_month_agent import MultiPolicyAgent
from agents.tuning_year_agent import TuningYearAgent

###################################################################
#####                Specify your agent here                  #####
###################################################################

# TuningAgent = MultiPolicyAgent
TuningAgent = MultiPolicyDayAgent
# TuningAgent = TuningYearAgent