from gym.envs.registration import register

register(
    id='carai-v0',
    entry_point='gym_carai.envs:caraiEnv',
)
register(
    id='carai-extrahard-v0',
    entry_point='gym_carai.envs:ExtraHardCarAIEnv',
)