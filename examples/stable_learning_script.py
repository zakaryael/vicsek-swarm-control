from parameters import *
from GymEnv import SwarmEnv
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import copy 
from utils import *
import pickle

# create the data folder if it does not exist
import os
if not os.path.exists("../data"):
    os.mkdir("../data")

# create a folder for the current experiment
experiment_name = time.strftime("%Y%m%d-%H%M%S")
experiment_path = os.path.join("../data", experiment_name)
os.mkdir(experiment_path)

# create the environment
env = SwarmEnv(N=N, L=L, v0=v0, r0=r0, eta=eta, potential_fields=potential_fields, Tmax=Tmax, target_radius=target_radius)
env = make_vec_env(SwarmEnv, n_envs=1, env_kwargs=dict(N=N, L=L, v0=v0, r0=r0, eta=eta, potential_fields=potential_fields, Tmax=Tmax, target_radius=target_radius))

# create the model
model = PPO("MlpPolicy", env, verbose=1)

# train the model 
n_total_timesteps = 100
save_frequency = 5
n_save = n_total_timesteps // save_frequency
# make a loop to save the model every n_save steps
for i in range(save_frequency):
    model.learn(n_save, reset_num_timesteps=False)
    model.save(os.path.join(experiment_path, f"_model_{i}"))
    # save the parameters of the model
    with open(os.path.join(experiment_path, f"params_{i}.txt"), "w") as f:
        f.write(str(model.get_parameters()))
    # evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    #save the mean reward and std reward in a file
    with open(os.path.join(experiment_path, f"eval_{i}.txt"), "w") as f:
        f.write(f"mean_reward: {mean_reward}, std_reward: {std_reward}")
    
    #save a the frames of the trained model's plots to make a gif later in a folder named "frames"
    if not os.path.exists(os.path.join(experiment_path, "images")):
        os.mkdir(os.path.join(experiment_path, "images"))
    save_frames(model, os.path.join(experiment_path, "images"))



# save a gif of the trained model's plots
save_learned_gif(model, os.path.join(experiment_path, "trained.gif"), fps=10)