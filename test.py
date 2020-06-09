import gym
import gym_carai
import time

env = gym.make('carai-simple-v0')
env.reset()

# Adaptive/Actor Critic Design
# x -> critic -> J
# x -> actor -> u

# function approximators:
#   Artificial neural networks

# nonlinear mapping, radial basis, feedforward
# Learning/training!  -> L3
# Sigmoidal 'always' works, might be suboptimal

# J(t) = r(t+1) + gamma J(t+1)
# TD = r(t+1) + gamma J(t+1) - J(t)
# get Temporal Difference to zero
#


# Critic:
# e_c (t) = J(t-1) - [gamma*J(t)-r(t)]
# E_c = 0.5*e_c^2
# Actor:
# e_a (t) = J(t) - J^*(t)     - J* = max value
# E_a = 0.5*e_c^2
# if r(x) < 0 all x, then J*=0

# J* difficult to find.
# Give all negative rewards, then you know J* = 0

# ADHDP Action Dependend Heuristic Dynamic Programming
# critic: find J from state u
# give u to critic, since plant not known. Otherwise
#           back propagation through plant, which is not known!
# update (both!) neural network weights based upon effect on error
#       gradient for a weight change.


# J* would be path length / velocity
# Watch Neural network lecture again!


start_time = time.time()
ep = 1
maxSteps = 10000
run = True
while run:
    print("--- starting run %s ---" % ep)
    run_time = time.time()
    env.reset()
    for i in range(10000):
        env.render('human')
        a = env.action_space.sample()
        obs, rewards, done, _, Terminate = env.step(a)  # take a random action
        obs = obs[2]
        if Terminate:
            run = False
        if done:
            break
    ep += 1
    print("--- %s seconds ---" % (time.time() - run_time))
    if not Terminate:
        '''Process epoch here'''
        t = 1
env.close()
print("--- total %s seconds ---" % (time.time() - start_time))






# Options:
# ADHDP (give u to critic)
# HDP (plant NN) (critic, actor AND plant Neural network training at the same time)

# current preference goes to ADHDP

"""
Error back propagation
Ea = 0.5 * ea ^2
ea(t) = J(t)-J*(t)
wak(t+1) = wak(t) + d wak(t)
d wak(t) = beta(t) * (- dEa/dWak)
dEa/dWak = dEa/dea  dea/dJ   dJ/du   du/dWak
dEa(t+1)/dWak = ...

"""