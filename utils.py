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
        obs, _, _ ,_ = env.step(action)
        img = env.render()
    imageio.mimsave(filename, images, fps=fps)

def save_learned_gif(model, filename, fps=30):
    # env is model.env
    images = []
    obs = model.env.reset()
    img = model.env.render()
    while not model.env.buf_dones:
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render()
    imageio.mimsave(filename, images, fps=fps)

def save_frames(model, folder):
    # env is model.env
    obs = model.env.reset()
    img = model.env.render()
    images = []
    images.append(img)
    
    while not model.env.buf_dones:
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render()
        images.append(img)
    #save the images list using pickle
    with open(os.path.join(folder, "images.pkl"), "wb") as f:
        pickle.dump(images, f)

        


# def make_gif_out_of_frames(folder, filename, fps=10):
#     images = []
#     for file in os.listdir(folder):
#         if file.endswith(".npy"):
#             images.append(np.load(os.path.join(folder, file)))
#     imageio.mimsave(filename, images, fps=fps)


def make_gif_out_of_frames(folder, filename, fps=10):
    #load the images list using pickle
    with open(os.path.join(folder, "images.pkl"), "rb") as f:
        images = pickle.load(f)
    imageio.mimsave(filename, images, fps=fps)
    




    