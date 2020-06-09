import gym
import gym_carai
import time

env = gym.make('carai-simple-v0')
env.reset()
start_time = time.time()

for _ in range(10000):
    arr = env.render('rgb_array')
    # a = -1 to 1
    a = env.action_space.sample()
    obs, rewards, done = env.step(a)  # take a random action
    if done:
        break

print("--- %s seconds ---" % (time.time() - start_time))

env.close()