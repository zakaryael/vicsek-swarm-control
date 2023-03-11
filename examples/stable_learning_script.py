from parameters import *
from GymEnv import SwarmEnv
from stable_baselines3 import PPO, A2C, SAC, TD3
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import copy 
from utils import *


# training parameters
n_total_timesteps = 100_000_000
save_frequency = 10
n_save = n_total_timesteps // save_frequency
n_eval = 1000
n_envs = 16


# create the data folder if it does not exist
import os
if not os.path.exists("../data"):
    os.mkdir("../data")

# create a folder for the current experiment
experiment_name = time.strftime("%Y%m%d-%H%M%S")
experiment_path = os.path.join("../data", experiment_name)
os.mkdir(experiment_path)



# create the environment
env = make_vec_env(SwarmEnv, n_envs=n_envs, env_kwargs=env_params)
eval_env = SwarmEnv(**env_params)

logs = os.path.join(experiment_path, "logs")
# create the model
model = PPO("MlpPolicy", env, verbose=1, gamma=1, tensorboard_log=logs)

#make a directory to save the logs

if not os.path.exists(logs):
    os.mkdir(logs)

# make a loop to save the model every n_save steps
for i in range(save_frequency):
    # train the model for n_save steps
    model.learn(n_save, reset_num_timesteps=False, tb_log_name='tb_log')
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
    # add the path to the environment parameters to save the environment state
    env_params["out_dir"] = env_dir
    eval_env = SwarmEnv(**env_params)
    obs = eval_env.reset()
    eval_env.render(mode=None)
    while not eval_env.done():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render(mode=None)

#make a gif of the trained model
save_learned_gif(model, eval_env, os.path.join(experiment_path, "trained_model.gif"), fps=10)