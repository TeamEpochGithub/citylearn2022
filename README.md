![Epoch Logo](https://drive.google.com/uc?export=view&id=1qzSSDQv3EQnSj7afgADCV8m3e9Zh2T0i)
![Citylearn Banner](https://images.aicrowd.com/uploads/ckeditor/pictures/906/content_Card_Banner.jpg)

# Team Epoch 
Our agent has the following policy:

![Equation](https://latex.codecogs.com/svg.image?\mathrm{pi}(O)&space;=&space;\frac{-\left(&space;\frac{\mathrm{LoadPredictor}\left(O\right)&space;-&space;\mathrm{SolarPredictor}\left(O\right)}{\left|{\mathrm{LoadPredictor}\left(O\right)&space;-&space;\mathrm{SolarPredictor}\left(O\right)}\right|}&space;\right)&space;\mathrm{PredictConsumption}\left(O\right)}{RemainingBatteryCapacity})

With O being our obserrvations. As you can see our agent predicts what the consumption will be like in future timesteps to change the action. It is able to predict like this as it records all the observations and learns to predict. In actuality however it only uses this prediction to check if there is going to be positive or negative consumption in the next step. Using this it picks the corresponding positive consumption or negative consumption policy.

# Table of Contents

- [Competition Overview](#competition-overview)
    + [Competition Phases](#competition-phases)
- [Getting Started](#getting-started)
- [How to write your own agent?](#how-to-write-your-own-agent)
- [Other Concepts](#other-concepts)
    + [Evaluation Metrics](#evaluation-metrics)
    + [Ranking Criteria](#ranking-criteria)
    + [Time constraints](#time-constraints)
  * [Local Evaluation](#local-evaluation)
  * [Contributing](#contributing)
  * [Contributors](#contributors)
- [Important links](#-important-links)


#  Competition Overview
[The CityLearn Challenge 2022](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge) focuses on the opportunity brought on by home battery storage devices and photovoltaics. It leverages [CityLearn](https://github.com/intelligent-environments-lab/CityLearn/tree/citylearn_2022), a Gym Environment for building distributed energy resource management and demand response. The challenge utilizes 1 year of operational electricity demand and PV generation data from 17 single-family buildings in the Sierra Crest home development in Fontana, California, that were studied for _Grid integration of zero net energy communities_.

Participants will develop energy management agent(s) and their reward function for battery charge and discharge control in each building with the goals of minimizing the monetary cost of electricity drawn from the grid, and the CO<sub>2</sub> emissions when electricity demand is satisfied by the grid.

### Competition Phases
The challenge consists of two phases: 
- In **Phase I**, the leaderboard will reflect the ranking of participants' submissions based on a 5/17 buildings training dataset.

- In **Phase II**, the leaderboard will reflect the ranking of participants' submissions based on an unseen 5/17 buildings validation dataset as well as the seen 5/17 buildings dataset. The training and validation dataset scores will carry 40% and 60% weights respectively in the Phase 2 score.

- In **Phase III**, participants' submissions will be evaluated on the 5/17 buildings training, 5/17 validation and remaining 7/17 test datasets. The training, validation and test dataset scores will carry 20%, 30% and 50% weights respectively in the Phase 3 score. The winner(s) of the competition will be decided using the leaderboard ranking in Phase III.



#  Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge).
3. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge-2022/citylearn-2022-starter-kit/-/forks/new) to create a fork.
4. **Clone** your forked repo and start developing your agent.
5. **Develop** your agent(s) following the template in [how to write your own agent](#how-to-write-your-own-agent) section.
5. **Develop** your reward function following the template in [how to write your own reward function](#how-to-write-your-own-reward-function) section.
6. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the citylearn simulator and report the metrics on the leaderboard of the competition.


# How to write your own agent?

We recommend that you place the code for all your agents in the `agents` directory (though it is not mandatory). You should implement the

- `register_reset`
- `compute_action`

**Add your agent name in** `user_agent.py`, this is what will be used for the evaluations.
  
Examples are provided in `agents/random_agent.py` and `agents/rbc_agent.py`.

To make things compatible with [PettingZoo](https://www.pettingzoo.ml/), a reference wrapper is provided that provides observations for each building individually (referred by agent id).

Add your agent code in a way such that the actions returned are conditioned on the `agent_id`. Note that different buildings can have different action spaces. `agents/orderenforcingwrapper.py` contains the actual code that will be called by the evaluator, if you want to bypass it, you will have to match the interfaces, but we recommend using the standard agent interface as shown in the examples.


# How to write your own reward function?
The reward function must be defined in `get_reward()` function in the [rewards.get_reward](rewards/get_reward.py) module. See [here](rewards/README.md) for instructions on how to define a custom reward function.


# Other Concepts
### Evaluation Metrics
Participants' submissions will be evaluated upon an equally weighted sum of two metrics at the aggregated district level where _district_ refers to the collection of buildings in the environment. The metrics include 1) district electricity cost, $`C_\textrm{entry}`$ and 2) district CO<sub>2</sub> emissions, $`G_\textrm{entry}`$ with the goal of minimizing the sum of each metric over the simulation period, $`t=0`$ to $`t=n`$ and $`e`$ episodes. The simulation period is 8,760 time steps i.e. one year, and participants can train on as many episodes of the simulation period, $`e`$, as needed.  $`C_\textrm{entry}`$ is bore by the individual buildings (customers) and $`G_\textrm{entry}`$ is an environmental cost. Each metric is normalized against those of the baseline where there is no electrical energy storage in batteries ($`C_\textrm{no battery}`$, $`G_\textrm{no battery}`$) such that values lower than that of the baseline are preferred.

```math
\textrm{score} = \frac{C_\textrm{entry}}{C_\textrm{no battery}} 
    + \frac{G_\textrm{entry}}{G_\textrm{no battery}}
```

### Ranking Criteria
Participants are ranked in ascending order of $`\textrm{score}`$ as the goal of the competition is to minimize $`\textrm{score}`$. 

In Phase 1, the training dataset score will carry 100% weight. By Phase 2, the training and validation dataset scores will carry 40% and 60% weights respectively. Finally in Phase 3, the training, validation and test dataset scores will carry 20%, 30% and 50% weights respectively .

The winner of each [phase](#competition-phases) will be the participant with the least weighted sum of scores from all considered datasets for the phase. In the event that multiple participants have the same $`\textrm{score}`$ in any of the phases, the ties will be broken in ascending order of agent complexity which, we interpret to be the simulation runtime.

### Time constraints

For Phase I, our agent should complete 5 episodes in 60 minutes. Note that the number of episodes and time can change depending on the phase of the challenge. However we will try to keep the throughput requirement of your agent, so you need not worry about phase changes. We only measure the time taken by your agent.



## Local Evaluation
- Participants can run the evaluation protocol for their agent locally with or without any constraint posed by the Challenge to benchmark their agents privately. See `local_evaluation.py` for details. You can change it as you like, it will not be used for the competition. You can also change the simulator schema provided under `data/citylearn_challenge_2022_phase_1/schema.json`, this will not be used for the competition.

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `agents/<your_agent>.py`.
- Import it in `user_agent.py`
- Test it out using `python local_evaluation.py`.
- Add any documentation for your approach at top of your file.
- Create merge request! 🎉🎉🎉 

## Contributors

- [Kingsley Nweye](https://www.aicrowd.com/participants/kingsley_nweye)
- [Zoltan Nagy](https://www.aicrowd.com/participants/nagyz)
- [Dipam Chakraborty](https://www.aicrowd.com/participants/dipam)

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge

- 🗣 Discussion Forum: https://discourse.aicrowd.com/c/neurips-2022-citylearn-challenge

- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge/leaderboards
