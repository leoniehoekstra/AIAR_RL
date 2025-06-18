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

# Monkey patch to skip nvidia-smi check
def _autodetect_num_gpus():
    return 1

resource_spec._autodetect_num_gpus = _autodetect_num_gpus

# ray.init(local_mode=True)


# plasma_dir = os.path.join(os.getcwd(), "tmp", "plasma")
# os.makedirs(plasma_dir, exist_ok=True)

# ray.init(
#     num_cpus=72,
#     _node_ip_address="127.0.0.1",  
#     ignore_reinit_error=True,
#     include_dashboard=False,
#     _temp_dir="tmp/ray",
#     _plasma_directory=plasma_dir
# )


# ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=r"..\tmp\ray")
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

# config = (
#     PPOConfig()
#     .environment(select_env)
#     .framework("torch")
#     .resources(
#         num_gpus=1  
#     )
#     .rollouts(
#         num_rollout_workers=0 
#     )
#     .training(
#         run_config=RunConfig(local_dir="./ray_results")
#     )
# )
# config = (
#     PPOConfig()
#     .environment(select_env)
#     .framework("torch")
#     .resources(num_gpus=1)
#     .rollouts(num_rollout_workers=65)
# )
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
      .environment(select_env)
      .framework("torch")
      # ---- hardware allocation -------------------------------------------
      .resources(
          num_gpus=1,              # learner uses the whole A16
          num_cpus_per_worker=1,   # one logical core per rollout worker
          num_gpus_per_worker=0    # workers stay CPU-only
      )
      .rollouts(num_rollout_workers=70)   # 70 workers + 1 driver = 71/72 cores
      # ---- training hyper-params -----------------------------------------
      .training(
          lr=1e-4,
          train_batch_size=280_000,     # 70 × 4000
          sgd_minibatch_size=32_768,    # ~8 minibatches / epoch
          num_sgd_iter=10,
          entropy_coeff=0.01,
          entropy_coeff_schedule=[[0, 0.01], [2e6, 0.0]],
          clip_param=0.2,
          kl_coeff=0.2,
          kl_target=0.01,
      )
      # .training(
      #     lr=1e-4,
      #     lr_schedule=[[0, 1e-4], [2e5, 5e-5]],     # new
      #     entropy_coeff=0.005,                      # new floor
      #     clip_param=0.15,                          # tighter clip
      #     kl_coeff=0.5,                             # stronger KL penalty
      #     sgd_minibatch_size=16_384,                # smaller MB
      #     num_sgd_iter=10,
      #     lambda_=0.95,                             # shorter GAE
      #     train_batch_size=280_000
      # )  
      # .training(
      #     lr=3e-4,
      #     entropy_coeff=0.005,                      # new floor
      #     kl_coeff=0.5,                             # stronger KL penalty
      #     sgd_minibatch_size=16_384,                # smaller MB
      #     num_sgd_iter=20,
      #     lambda_=0.95,                             # shorter GAE
      #     train_batch_size=280_000
      # )  
)

    
# config = PPOConfig().resources(num_gpus=1).rollouts(num_rollout_workers=24).framework("torch")

agent = config.build()

# agent = config.build(env=select_env)
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
