import imageio as imageio
import numpy as np
import os
import pickle


def save_gif(env, filename, timesteps=1000, fps=30):
    images = []
    obs = env.reset()
    img = env.render()
    for i in range(timesteps):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        img = env.render()
    imageio.mimsave(filename, images, fps=fps)


def save_learned_gif(model, env, filename, fps=30):
    # env is model.env
    images = []
    obs = env.reset()
    img = env.render()
    while not env.done():
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        img = env.render()
    imageio.mimsave(filename, images, fps=fps)


def save_frames(model, env, folder, fname="images.pkl"):
    # env is model.env
    obs = env.reset()
    img = env.render()
    images = []
    images.append(img)

    while not env.done():
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        img = env.render()
        images.append(img)
    # save the images list using pickle
    with open(os.path.join(folder, fname), "wb") as f:
        pickle.dump(images, f)


def make_gif_out_of_frames(path, filename, fps=10):
    # load the images list using pickle
    with open(path, "rb") as f:
        images = pickle.load(f)
    imageio.mimsave(filename, images, fps=fps)
