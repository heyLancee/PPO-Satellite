import gym
import pandas as pd
import torch

from DynamicNet import AttitudeDynamicsNN
import DynamicNet
import TD3
from satellite import *


def eval_pid(pid, env_name, seed, path=None, is_plot=False):
    if env_name == "Satellite":
        eval_env = Satellite()
    elif env_name == "FaultSatellite":
        eval_env = FaultSatellite()
    elif env_name == "SunPointSatellite":
        eval_env = SunPointSatellite()
    elif env_name == "SunPointFaultSatellite":
        eval_env = SunPointFaultSatellite()
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(seed + np.random.randint(1, 100))

    rewards = []
    states = []
    actions = []
    state, done = eval_env.reset(), False

    eval_env.fault_mode = 0

    while not done:
        pid_action = pid.select_action(eval_env.sb, eval_env.sd, eval_env.omega)

        action = np.linalg.pinv(eval_env.C)@pid_action
        state, reward, done, _ = eval_env.step(action.reshape(-1, 1))

        states.append(state)
        rewards.append(reward)
        actions.append(action)

    # 在循环结束后转换为NumPy数组
    states = np.array(states)
    rewards = np.array(rewards)
    actions = np.array(actions)

    print("---------------------------------------")
    print(f"reward: {np.mean(rewards)}")
    print("---------------------------------------")

    if path is not None:
        df = pd.DataFrame(states, columns=[f'state_{i}' for i in range(len(states[0]))])
        df_uc = pd.DataFrame(actions, columns=[f'u_{i}' for i in range(len(actions[0]))])
        df = pd.concat([df, df_uc], axis=1)
        df['reward'] = rewards
        df.to_csv(path, index=False)

    if is_plot:
        eval_env.plot()

    return np.mean(rewards)


def eval_policy(agent, dynamic_net, env_name, fault_mode, seed, path=None, is_plot=False):
    if env_name == "Satellite":
        eval_env = Satellite()
    elif env_name == "FaultSatellite":
        eval_env = FaultSatellite()
    elif env_name == "SunPointSatellite":
        eval_env = SunPointSatellite()
    elif env_name == "SunPointFaultSatellite":
        eval_env = SunPointFaultSatellite()
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(seed)

    rewards = []
    states = []
    actions = []
    state, done = eval_env.reset(), False
    if fault_mode != -1:
        eval_env.fault_mode = fault_mode
    state = np.concatenate((state, np.zeros(DynamicNet.OUTPUT_NUM)))

    while not done:
        if agent is not None:
            agent_action = agent.select_action(np.array(state))
        else:
            agent_action = np.zeros(4)
        action = np.diag(agent_action) @ eval_env.u_max
        
        # dynamic net
        net_input = np.concatenate((eval_env.omega.flatten(), (eval_env.C@action).flatten()))
        pred = dynamic_net(torch.tensor(net_input, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()

        next_state, reward, done, _ = eval_env.step(action.reshape(-1, 1))
        
        pred_error = eval_env.omega.flatten() - pred.flatten()

        next_state = np.concatenate((next_state.flatten(), pred_error.flatten()))
        state = next_state

        states.append(state)
        rewards.append(reward)
        actions.append(action)

    # 在循环结束后转换为NumPy数组
    states = np.array(states)
    rewards = np.array(rewards)
    actions = np.array(actions)

    # print("---------------------------------------")
    # print(f"reward: {np.mean(rewards)}")
    # print("---------------------------------------")

    if path is not None:
        df = pd.DataFrame(states, columns=[f'state_{i}' for i in range(len(states[0]))])
        df_uc = pd.DataFrame(actions, columns=[f'u_{i}' for i in range(len(actions[0]))])
        df = pd.concat([df, df_uc], axis=1)
        df['reward'] = rewards
        df.to_csv(path, index=False)

    if is_plot:
        eval_env.plot()

    return np.sum(rewards)


def plot_result(data: pd.DataFrame):
    # 提取四元数和角速度
    quaternions = data[['state_0', 'state_1', 'state_2', 'state_3']]
    angular_velocities = data[['state_4', 'state_5', 'state_6']]

    u_c = data[['u_0', 'u_1', 'u_2', 'u_3']]

    # 绘制图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # 绘制四元数
    quaternions.plot(ax=ax1)
    ax1.set_title('Quaternions')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend(['q0', 'q1', 'q2', 'q3'])

    # 绘制角速度
    angular_velocities.plot(ax=ax2)
    ax2.set_title('Angular Velocities')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value (rad/s)')
    ax2.legend(['wx', 'wy', 'wz'])

    # 绘制角速度
    u_c.plot(ax=ax3)
    ax3.set_title('Torque')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Value (Nm)')
    ax3.legend(['u1', 'u2', 'u3', 'u4'])

    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    policy = "TD3"
    seed = np.random.randint(1, 100)
    # seed = 2
    env_name = "SunPointFaultSatellite"
    dynamic_net_path = "models/dynamic_net/attitude_dynamics_model.pth"
    hidden_size = [64, 128]
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    policy_model_path = "2024-12-26_21-56-43_1\TD3_SunPointFaultSatellite_1"

    if env_name == "Satellite":
        env = Satellite()
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

    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    state_dim += TD3.STATE_APPEND_NUM
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
    }

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = TD3.TD3(**kwargs)
    else:
        raise NotImplementedError

    if policy_model_path != "":
        policy.load(f"./models/{policy_model_path}")

    dynamicNet = DynamicNet.AttitudeDynamicsNN(hidden_size)
    if dynamic_net_path != "":
        print(f"Load dynamic net from {dynamic_net_path}")
        dynamicNet.load_model(dynamic_net_path)

    # Evaluate untrained policy
    path = "results/eval_res.csv"
    eval_policy(policy, dynamicNet, env_name, 1, seed, path, is_plot=True)
