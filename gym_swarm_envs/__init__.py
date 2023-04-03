from gym.envs.registration import register

register(
    id="vicsek-swarm-v0",
    entry_point="gym-envs.envs:SwarmEnv",
)
