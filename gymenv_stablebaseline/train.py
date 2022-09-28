import sys

from gymenv import CityLearnEnv
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


env = CityLearnEnv()
obs = env.reset()

# check_env(env)
model = PPO('MlpPolicy', env, verbose=0, n_steps=512).learn(total_timesteps=8760)
model.save("../gymenv_stablebaseline/data/ppo_test_steps512")


obs = env.reset()
actions = []
rewards = []

epochs = 10

# for j in range(epochs):

for i in range(8760):
    action, _state = model.predict(obs, deterministic=True)
    # print(action)
    actions.append(action)

    obs, reward, done, info = env.step(action)
    print(reward)

    rewards.append(reward)

    # env.render()
    if done:
        obs = env.reset()

print(sum(rewards))
# print(actions)