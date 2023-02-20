import imageio

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
    