import gym
import gym_carai
import time
import numpy as np

# Setup for training etc.
env = gym.make('carai-simple-v0')  # First open the environment

observation_shape = env.observation_space.shape[0]

start_time = time.time()           # Register current time
epoch = 1                          # Current episode
maxEpoch = 100                     # max amount of epochs
maxEpochTime = 500                 # [s] max seconds to spend per epoch
dt = 1/60                          # fps (should equal monitor refresh rate)
maxSteps = int(maxEpochTime/dt)    # max duration of an epoch
Terminate = None
done = 0
learning_rate = 0.001
run = True
maxRewardSoFar = -90000

rewardavglist = []
criticavglist = []

while run:
    print("--- starting run %s ---" % epoch)
    run_time = time.time()
    env.reset()

    # initial values
    epoch_loss = 0
    action = np.array([0])

    for i in range(maxSteps):
        # calculate next step
        env.render('manual')  # manual, human, rgb_array
        obs, reward, done, info_dict, Terminate = env.step(action, dt)
        if Terminate:  # Window was closed.
            epoch = maxEpoch*2
            run = False
        if done:
            break
    print("--- %s seconds ---" % (time.time() - run_time))

    epoch += 1
env.close()
print("--- total %s seconds ---" % (time.time() - start_time))
