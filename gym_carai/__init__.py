from gym.envs.registration import register

register(
    id='carai-v0',
    entry_point='gym_carai.envs:CarAIEnv',
)
register(
    id='carai-simple-v0',
    entry_point='gym_carai.envs:SimpleCarAIEnv',
)
register(
    id='carai-less-simple-v0',
    entry_point='gym_carai.envs:LessSimpleCarAIEnv',
)