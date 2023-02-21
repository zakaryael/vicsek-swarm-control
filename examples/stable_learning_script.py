from parameters import *
from GymEnv import SwarmEnv
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import copy 
from utils import *


# training parameters
n_total_timesteps = 2
save_frequency = 2
n_save = n_total_timesteps // save_frequency
n_eval = 1
n_envs = 1


# create the data folder if it does not exist
import os
if not os.path.exists("../data"):
    os.mkdir("../data")

# create a folder for the current experiment
experiment_name = time.strftime("%Y%m%d-%H%M%S")
experiment_path = os.path.join("../data", experiment_name)
os.mkdir(experiment_path)

# create the environment
env = make_vec_env(SwarmEnv, n_envs=n_envs, env_kwargs=dict(N=N, L=L, v0=v0, r0=r0, eta=eta, potential_fields=potential_fields, Tmax=Tmax, target_radius=target_radius))
eval_env = SwarmEnv(N=N, L=L, v0=v0, r0=r0, eta=eta, potential_fields=potential_fields, Tmax=Tmax, target_radius=target_radius)


# create the model
model = PPO("MlpPolicy", env, verbose=0, gamma=1)

#make a directory to save the logs 
logs = os.path.join(experiment_path, "logs")
if not os.path.exists(logs):
    os.mkdir(logs)

# make a loop to save the model every n_save steps
for i in range(save_frequency):
    # train the model for n_save steps
    model.learn(n_save, reset_num_timesteps=False, tb_log_name=logs)
    # save the model
    model.save(os.path.join(experiment_path, f"_model_{i}"))

    # evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval)
    #save the mean reward and std reward in a file
    with open(os.path.join(experiment_path, f"eval_{i}.txt"), "w") as f:
        f.write(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
    
    #run the model on the eval env and save the env state
    env_dir = os.path.join(experiment_path, f"env_{i}")
    if not os.path.exists(env_dir):
        os.mkdir(env_dir)
    eval_env = SwarmEnv(N=N, L=L, v0=v0, r0=r0, eta=eta, potential_fields=potential_fields, Tmax=Tmax, target_radius=target_radius, out_dir=env_dir)
    obs = eval_env.reset()
    eval_env.render()
    while not eval_env.done():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()

# #make a gif of the trained model
# save_learned_gif(model, eval_env, os.path.join(experiment_path, "trained_model.gif"), fps=10)