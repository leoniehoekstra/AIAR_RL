import os
import shutil
import socket
import pathlib
from pathlib import Path
import ray
from ray.air import RunConfig

from ray.tune.registry import register_env
from explore_agent.envs.exploring_gym import ExploreDrone
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray._private import resource_spec

def _autodetect_num_gpus():
    return 1

resource_spec._autodetect_num_gpus = _autodetect_num_gpus

ray.init(
    _node_ip_address="127.0.0.1", # otherwise it crashes on the university serverrr
    include_dashboard=False,
    ignore_reinit_error=True,
    _temp_dir=str(Path.home() / "tmp" / "ray"),
)


chkpt_root = "tmp/ppo/checkpoint_{}"
ray_results = "{}/ray_results/".format(os.getenv("HOME"))


select_env = "ExploreAgent-v0"
register_env(select_env, lambda config: ExploreDrone({"env_name": "playground",  "reward_mode": "continuous"}))

from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
      .environment(select_env)
      .framework("torch")
      .resources(
          num_gpus=1,
          num_cpus_per_worker=1,
          num_gpus_per_worker=0
      )
      .rollouts(num_rollout_workers=70)
      .training(
          lr=1e-4,
          train_batch_size=280_000,
          sgd_minibatch_size=32_768,
          num_sgd_iter=10,
          entropy_coeff=0.01,
          entropy_coeff_schedule=[[0, 0.01], [2e6, 0.0]],
          clip_param=0.2,
          kl_coeff=0.2,
          kl_target=0.01,
      )
)

agent = config.build()
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f}"
best_reward = float("-inf")
best_checkpoint = None

n_iter = 300
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
