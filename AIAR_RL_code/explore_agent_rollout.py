# import gym
import gymnasium as gym
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_USE_CUSTOM_LOGGING"] = "0"
import shutil
import ray
from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
import ray.rllib.algorithms.ppo.ppo as ppo
from ray.rllib.algorithms.ppo.ppo import PPOConfig
import ray.rllib.algorithms.sac.sac as sac
from ray.rllib.algorithms.sac.sac import SACConfig
from ray._private import resource_spec
# Monkey patch to skip nvidia-smi check
def _autodetect_num_gpus():
    return 0

resource_spec._autodetect_num_gpus = _autodetect_num_gpus
# ray.init(ignore_reinit_error=True, num_cpus=1, include_dashboard=False, _temp_dir=r"D:\Workspace\UT_postdoc\10_teaching\10_MSc_Robotics\2024_2025\AIAR\practical_assignment\tmp\ray")
ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=r"D:\Workspace\UT_postdoc\10_teaching\10_MSc_Robotics\2024_2025\AIAR\practical_assignment\tmp\ray")
from time import sleep

chkpt_root = "tmp/ppo"
# chkpt_root = "tmp/sac"
# ray_results = "{}/ray_results/".format(os.getenv("HOME"))
ray_results = r"D:\Workspace\UT_postdoc\10_teaching\10_MSc_Robotics\2024_2025\AIAR\practical_assignment\tmp\ray"



select_env = "ExploreAgent-v0"
# register_env(select_env, lambda config: ExploreDrone())
register_env(select_env, lambda config: ExploreDrone({"env_name": "playground"}))
# config = PPOConfig().resources(num_gpus=0).rollouts(num_rollout_workers=1).framework("torch")
config = PPOConfig().resources(num_gpus=1).rollouts(num_rollout_workers=1).framework("torch")

config["log_level"] = "WARN"
agent = ppo.PPO(config, env=select_env)
chkpt_file = "tmp/ppo/checkpoint_best"

# agent.restore(chkpt_file)
agent = agent.from_checkpoint(chkpt_file)
env = gym.make(select_env)
state,_ = env.reset()

sum_reward = 0
n_step = 1000
for step in range(n_step):
    action = agent.compute_single_action(state)
    # sleep(0.0416)
    sleep(0.1)
    # sleep(0.5)
    state, reward, done, _,info = env.step(action)
    sum_reward += reward
    env.render()
    img = env.render(mode='rgb_array')
    print(img)
    if done == 1:
        print("wall hit!!!")
        print("cumulative reward", sum_reward)
        state = env.reset()
        sum_reward = 0
        break
