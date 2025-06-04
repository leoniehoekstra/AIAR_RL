import os
import shutil
import ray

from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray._private import resource_spec
# Monkey patch to skip nvidia-smi check
def _autodetect_num_gpus():
    return 0

resource_spec._autodetect_num_gpus = _autodetect_num_gpus
ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=r"..\tmp\ray")


chkpt_root = "tmp/ppo/checkpoint_{}"
ray_results = "{}/ray_results/".format(os.getenv("HOME"))


select_env = "ExploreAgent-v0"
register_env(select_env, lambda config: ExploreDrone({"env_name": "playground",  "reward_mode": "continuous"}))
config = PPOConfig().resources(num_gpus=1).rollouts(num_rollout_workers=24).framework("torch")



agent = config.build(env=select_env)
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}"
best_reward = float("-inf")
best_checkpoint = None

n_iter = 500
save_interval = 10
warmpup_iterations = 50

for n in range(n_iter):
    result = agent.train()
    if n > warmpup_iterations:
        if result["episode_reward_mean"] > best_reward:
                best_reward = result["episode_reward_mean"]
                if best_checkpoint:
                    shutil.rmtree(chkpt_root.format("best"), ignore_errors=True, onerror=None)
                best_checkpoint = agent.save(chkpt_root.format("best"))
                print("Iteration {}: New Best Reward {:.2f}".format(
                n + 1,
                result["episode_reward_mean"],
            ))
                
    if n%save_interval==0:
        chkpt_file = agent.save(chkpt_root.format(n))

    print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"]
            ))
