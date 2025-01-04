import os
import glob
import time
import argparse
from datetime import datetime

import torch
import numpy as np

import gym
from DynamicNet import *
from PPO import PPO
from satellite import *

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "Satellite"
    has_continuous_action_space = True
    max_ep_len = 2000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    hidden_dim = 256

    dyn_hidden_size = [64, 128]
    dyn_net_path = ""

    #####################################################

    if env_name == "Satellite":
        env = Satellite()
    elif env_name == "Satellite":
        env = Satellite()
    elif env_name == "FaultSatellite":
        env = FaultSatellite()
    elif env_name == "SunPointSatellite":
        env = SunPointSatellite()
    elif env_name == "SunPointFaultSatellite":
        env = SunPointFaultSatellite()
    else:
        env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]
    state_dim += OUTPUT_NUM

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    dynamic_net = AttitudeDynamicsNN(dyn_hidden_size)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # file exists
    if os.path.exists(checkpoint_path):
        print("loading network from : " + checkpoint_path)
        ppo_agent.load(checkpoint_path)

    if dyn_net_path != "":
        print(f"Load dynamic net: {dyn_net_path}")
        dynamic_net.load_model(dyn_net_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()
        state = np.concatenate((state.flatten(), np.zeros(OUTPUT_NUM)), axis=0)

        # for t in range(1, max_ep_len+1):
        while True:
            action = ppo_agent.select_action(state)
            action = np.diag(action) @ env.u_max

            # dynamic net
            net_input = np.concatenate((env.omega.flatten(), (env.C@action).flatten()))
            pred = dynamic_net(torch.tensor(net_input, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()
            state, reward, done, _ = env.step(action)
            pred_error = env.omega.flatten() - pred.flatten()
            state = np.concatenate((state.flatten(), pred_error.flatten()))

            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
