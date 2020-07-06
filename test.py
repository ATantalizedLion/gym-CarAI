import gym
import gym_carai
import time
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # this line forces to run on CPU
import tensorflow as tf
# print(tf.config.experimental.list_physical_devices('GPU') )  # show all gpus
# print(tf.__version__)  # show tf version


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class ActorCriticModel(tf.keras.Model):
    def __init__(self, learning_rate):
        super(ActorCriticModel, self).__init__()

        # actor part of Model (policies)
        self.inner1 = tf.keras.layers.Dense(200, activation='relu')
        self.turning_logits = tf.keras.layers.Dense(3)  # 3 logits for turning policy: left/straight/right

        # critic part of model (value function)
        self.inner2 = tf.keras.layers.Dense(200, activation='relu')
        self.value = tf.keras.layers.Dense(1)  # condense back into 1

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate, use_locking=True)

    def call(self, inputs):
        x = self.inner1(inputs)
        logits = self.turning_logits(x)
        v1 = self.inner2(inputs)
        J  = self.value(v1)
        return logits, J

def get_loss(model, memory, done, gamma=0.99):

    # continous version of Belmann:
    reward_sum = 0
    discounted_rewards = []
    for reward in memory.rewards[::-1]:
        reward_sum = reward + gamma * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards = discounted_rewards[::-1]

    # get J for each timestep
    logits, values = model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                     dtype=tf.float32) - values

    critic_loss = (advantage) ** 2

    # actor Loss:

    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
    actor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
    actor_loss *= tf.stop_gradient(advantage)

    actor_loss -= 0.01 * entropy

    total_loss = tf.reduce_mean((0.5 * critic_loss + actor_loss))

    print("critic loss -- %s" % (critic_loss.numpy()))
    print("actor loss -- %s" % (actor_loss.numpy()))
    print("total loss -- %s" % (total_loss.numpy()))
    return total_loss


# Setup for training etc.
env = gym.make('carai-simple-v0')  # First open the environment
start_time = time.time()           # Register current time
epoch = 1                          # Current episode
maxEpoch = 100                     # max amount of epochs
maxEpochTime = 50                  # [s] max seconds to spend per epoch
dt = 1/60                          # fps (should equal monitor refresh rate)
maxSteps = int(maxEpochTime/dt)    # max duration of an epoch
Terminate = None
done = 0
learning_rate = 0.0001
mem = Memory()
run = True


Model = ActorCriticModel(learning_rate)  # global network
Model(tf.convert_to_tensor(np.random.random((1, 1)), dtype=tf.float32))

while run:
    print("--- starting run %s ---" % epoch)
    run_time = time.time()
    env.reset()
    mem.clear()

    # initial values
    obs = np.array([[500]])
    epoch_loss = 0

    for i in range(maxSteps):
        # calculate next step
        env.render('human')  # manual, human, rgb_array

        logits, _ = Model(obs)
        probs = tf.nn.softmax(logits)
        action = np.array([np.random.choice(3, p=probs.numpy()[0])])  # returns 0,1,2, action space = -1 to 1
        obs, reward, done, info_dict, Terminate = env.step(action, dt)
        mem.store(obs, action[0], reward)
        if Terminate:  # Window was closed.
            epoch = maxEpoch*2
            run = False
        if done:
            break
    if not Terminate:
        # GradientTape tracks the gradient of all variables within scope
        with tf.GradientTape() as tape:
            total_loss = get_loss(Model, mem, done)
        epoch_loss += total_loss
        # Apply found gradients to model
        grads = tape.gradient(total_loss, Model.trainable_weights)
        Model.opt.apply_gradients(zip(grads, Model.trainable_weights))
    print("--- %s seconds ---" % (time.time() - run_time))

    epoch += 1

env.close()
print("--- total %s seconds ---" % (time.time() - start_time))

