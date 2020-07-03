import gym
import gym_carai
import time
import numpy as np
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf


# from tensorflow.python.client import device_lib
# device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
# if device_name[0] == "/device:GPU:0":
#     device_name = "/gpu:0"
#     print('GPU')
# else:
#     device_name = "/cpu:0"
#     print('CPU')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


'''
# J(t) = r(t+1) + gamma J(t+1)
# TD = r(t+1) + gamma J(t+1) - J(t)
# get Temporal Difference to zero


# ADHDP Action Dependend Heuristic Dynamic Programming
# give u to critic, since plant not known. Otherwise
#           back propagation through plant, which is not known!
# update (both!) neural network weights based upon effect on error
#       gradient for a weight change.


# J* would be path length / velocity
# Watch Neural network lecture again!


# Set up the models




# u = steering direction [-1 to 1]
# x = distance to wall right side
'''

# Plant NN:
# Find x from state u

# Critic, task is to find J from state u
# e_c (t) = J(t-1) - [gamma*J(t)-r(t)]
# E_c = 0.5*e_c^2
# if r(x) < 0 all x, then J*=0    #r(x) = reward


# Actor, task is to reach a goal value J*
# e_a (t) = J(t) - J^*(t)     - J* = max value
# E_a = 0.5*e_c^2
# if r(x) < 0 all x, then J*=0
# J* difficult to find.
# Give all negative rewards, then you know J* = 0

u_size = 1
x_size = 1

# Actor
# x in, u out, 2 hidden layers with 10 neurons.

'''
# Critic
# x in, estimated J out, 2 hidden layers with 10 neurons.
critic = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(x_size,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(u_size)
])
# Plant NN
# u in, estimated x out,  2 hidden layers with 10 neurons.
plant = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(u_size,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(x_size)
])
'''

# compile models

inputs = tf.keras.Input(shape=(1,))
x = tf.keras.layers.Dense(5, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(x)
actor = tf.keras.Model(inputs=inputs, outputs=outputs)

actor.compile()
actor.summary()

modelinput = np.array([1])
print(actor.predict(modelinput))


# Start training
env = gym.make('carai-simple-v0')  # First open environment
start_time = time.time()  # Register current time
ep = 1  # Current episode
maxSteps = 10000  # max duration epoch
dt = 1/60


run = True
while run:
    print("--- starting run %s ---" % ep)
    run_time = time.time()
    env.reset()
    obs = np.array([40])
    for i in range(10000):
        # env.render('human')
        action = actor.predict(obs)
        action = action[0]
        obs, rewards, done, info_dict, Terminate = env.step(action, dt)  # take a random action
        if Terminate:  # Window was closed.
            run = False
            break
        if done:
            break
    ep += 1
    print("--- %s seconds ---" % (time.time() - run_time))
    if not Terminate:
        '''Process epoch here'''
        t = 1
env.close()
print("--- total %s seconds ---" % (time.time() - start_time))

