#!/usr/bin/env python3.9

import gym
import tinycarlo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src import action_noise
from src import buffer
from src import models

env = gym.make("tinycarlo-v0", fps=30, camera_resolution=(480//4,640//4), reward_red='done')

state_shape = env.observation_space.shape # 3 channel image
num_actions = env.action_space.shape[0]

lower_bound = -1 # min action value
upper_bound = 1 # max action value

def policy(state, noise_object):
    '''
    returns an action sampled from Actor network plus some noise for exploration.
    '''
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


std_dev = 0.2
ou_noise = action_noise.OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = models.get_actor(state_shape=state_shape)
critic_model = models.get_critic(state_shape=state_shape,num_actions=num_actions)

target_actor = models.get_actor(state_shape=state_shape)
target_critic = models.get_critic(state_shape=state_shape, num_actions=num_actions)

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = buffer.Buffer(state_shape, num_actions, actor_model, target_actor, critic_model, target_critic, 50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0
    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            observation = env.reset()
            break
        prev_state = state
    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

env.close()

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

