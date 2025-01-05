import os
import glob
import time
import argparse
from datetime import datetime

import torch
import numpy as np

import gym

from PPO import PPO
from satellite import *
from DynamicNet import *

################################### Training ###################################
def train():
    print("============================================================================================")
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_num", type=int)
    parser.add_argument("--env_name", default="Satellite", type=str)
    parser.add_argument("--has_continuous_action_space", default=True, type=bool)
    parser.add_argument("--max_ep_len", default=2000, type=int)
    parser.add_argument("--max_training_timesteps", default=int(3e6), type=int)
    parser.add_argument("--save_model_freq", default=int(1e5), type=int)
    parser.add_argument("--print_freq_factor", default=10, type=int)
    parser.add_argument("--log_freq_factor", default=2, type=int)
    parser.add_argument("--action_std", default=0.6, type=float)
    parser.add_argument("--action_std_decay_rate", default=0.05, type=float)
    parser.add_argument("--min_action_std", default=0.1, type=float)
    parser.add_argument("--action_std_decay_freq", default=int(2.5e5), type=int)
    parser.add_argument("--update_timestep_factor", default=4, type=int)
    parser.add_argument("--K_epochs", default=80, type=int)
    parser.add_argument("--eps_clip", default=0.2, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lr_actor", default=0.0003, type=float)
    parser.add_argument("--lr_critic", default=0.001, type=float)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dyn_hidden_size", default=[64, 128], type=int, nargs='+')
    parser.add_argument("--dyn_net_path", default="", type=str)

    args = parser.parse_args()

    ####### initialize environment hyperparameters ######
    env_name = args.env_name

    has_continuous_action_space = args.has_continuous_action_space  # continuous action space; else discrete

    max_ep_len = args.max_ep_len                # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * args.print_freq_factor        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * args.log_freq_factor           # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_model_freq          # save model frequency (in num timesteps)

    action_std = args.action_std                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = args.action_std_decay_rate        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = args.min_action_std                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = args.action_std_decay_freq  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * args.update_timestep_factor      # update policy every n timesteps
    K_epochs = args.K_epochs               # update policy for K epochs in one PPO update

    eps_clip = args.eps_clip          # clip parameter for PPO
    gamma = args.gamma            # discount factor

    lr_actor = args.lr_actor       # learning rate for actor network
    lr_critic = args.lr_critic       # learning rate for critic network

    random_seed = args.random_seed         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    if env_name == "Satellite":
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

    hidden_dim = args.hidden_dim

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = args.seq_num

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = args.seq_num      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    dynamic_net = AttitudeDynamicsNN(args.dyn_hidden_size)
    if args.dyn_net_path != "":
        print(f"Load dynamic net: {args.dyn_net_path}")
        dynamic_net.load_model(args.dyn_net_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        state = np.concatenate([state, np.zeros(OUTPUT_NUM)], axis=0)

        # for t in range(1, max_ep_len+1):
        while True:

            # select action with policy
            action = ppo_agent.select_action(state)
            action = np.diag(action) @ env.u_max

            # dynamic net
            net_input = np.concatenate((env.omega.flatten(), (env.C@action).flatten()))
            pred = dynamic_net(torch.tensor(net_input, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()

            state, reward, done, _ = env.step(action)
            pred_error = env.omega.flatten() - pred.flatten()
            state = np.concatenate((state.flatten(), pred_error.flatten()))

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    
