import random

import numpy as np

from satellite_func import *
from gym import spaces
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


class Satellite:
    def __init__(self, t_max=200, ts=0.1):
        if hasattr(self, 'is_init') and self.is_init:
            print("Already initialized, skipping initialization.")
            return

        self.is_init = False
        self.ts = ts
        self.t = 0
        self.t_max = t_max
        self._max_episode_steps = int(self.t_max / self.ts)

        self.j = np.array([
            [12, 0, 0],
            [0, 15, 0],
            [0, 0, 18]
        ])
        self.j_inv = np.linalg.inv(self.j)
        self.C = np.array([
            [1, 0, 0, 3 ** 0.5 / 3],
            [0, 1, 0, 3 ** 0.5 / 3],
            [0, 0, 1, 3 ** 0.5 / 3]
        ])
        self.omega_buffer = []
        self.q_buffer = []
        self.u_buffer = []  # 单机输出力矩
        self.qe_buffer = []
        self.omega_e_buffer = []
        self.state = None
        self.q = None
        self.omega = None
        self.qd = None
        self.u_max = np.array([0.05, 0.05, 0.05, 0.05])

        obs = np.array([1, 1, 1, 1, 5, 5, 5], dtype=np.float32)
        action = np.array([1, 1, 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(-action, action, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

    def step(self, torque):
        torque = torque.reshape(-1, 1)
        # clip
        torque = np.clip(torque.flatten(), -self.u_max, self.u_max)
        u = (self.C @ torque).reshape(-1, 1)

        self.q, self.omega = R_K(self.q, self.omega, self.ts, self.j_inv, self.j, u)

        omega_d = get_omega_d(self.t)
        qe = get_q_e(self.qd, self.q)
        qev = qe[1:]
        omega_e = get_omega_e(self.omega, omega_d, qe)
        self.state = np.concatenate([qe, omega_e], axis=0).flatten()

        self.omega_buffer.append(self.omega.flatten())
        self.q_buffer.append(self.q.flatten())
        self.u_buffer.append(torque.flatten())
        self.qe_buffer.append(qe.flatten())
        self.omega_e_buffer.append(omega_e.flatten())
        reward = self.reward(torque, qev, omega_e)

        self.t += self.ts
        done = False
        if self.t >= self.t_max:
            done = True
        return self.state, reward, done, {}

    def reward(self, f, qev, omega_e):
        reward_1 = 0
        reward_2 = -4 * np.linalg.norm(f)
        reward_3 = -20 * np.linalg.norm(qev)
        reward_4 = -10 * np.linalg.norm(omega_e)
        reward = reward_1 + reward_2 + reward_3 + reward_4
        return reward

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return seed
    
    def plot(self):
        qe_buffer = np.array(self.qe_buffer)
        omega_e_buffer = np.array(self.omega_e_buffer) * 180 / np.pi
        u_buffer = np.array(self.u_buffer)
        q_buffer = np.array(self.q_buffer)
        omega_buffer = np.array(self.omega_buffer)

        times = np.linspace(0, self.t_max, len(qe_buffer))

        # qe_buffer
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(times, qe_buffer[:, 0], label='qe0')
        ax.plot(times, qe_buffer[:, 1], label='qe1')
        ax.plot(times, qe_buffer[:, 2], label='qe2')
        ax.plot(times, qe_buffer[:, 3], label='qe3')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Quaternion') 

        # omega_e_buffer
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(times, omega_e_buffer[:, 0], label='omega_e0')
        ax.plot(times, omega_e_buffer[:, 1], label='omega_e1')
        ax.plot(times, omega_e_buffer[:, 2], label='omega_e2')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Omega_e')

        # u_buffer
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(times, u_buffer[:, 0], label='ux')
        ax.plot(times, u_buffer[:, 1], label='uy')
        ax.plot(times, u_buffer[:, 2], label='uz')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Torque')

        # q_buffer
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(times, q_buffer[:, 0], label='q0')
        ax.plot(times, q_buffer[:, 1], label='q1')
        ax.plot(times, q_buffer[:, 2], label='q2')
        ax.plot(times, q_buffer[:, 3], label='q3')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Quaternion')
        
        # omega_buffer
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(times, omega_buffer[:, 0], label='omega0')
        ax.plot(times, omega_buffer[:, 1], label='omega1')
        ax.plot(times, omega_buffer[:, 2], label='omega2')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Omega')
     
        plt.show()

    def reset(self):
        self.qd = np.array([[1], [0], [0], [0]])
        omega_d = get_omega_d(self.t)
        self.qd = np.random.random((4, 1))
        self.qd = self.qd / np.linalg.norm(self.qd)
        self.q = np.random.random((4, 1))
        # self.q = np.array([[1], [2], [3], [4]])
        self.q = self.q / np.linalg.norm(self.q)
        self.omega = (2 * np.random.random((3, 1)) - 1) * 0.2
        # self.omega = np.array([[-0.3], [0.1], [0.2]])
        qe = get_q_e(self.qd, self.q)
        omega_e = get_omega_e(self.omega, omega_d, qe)
        self.state = np.concatenate([qe, omega_e], axis=0).flatten()
        self.t = 0
        self.q_buffer = []
        self.omega_buffer = []
        self.u_buffer = []
        self.qe_buffer = []
        self.omega_e_buffer = []

        # print("quat init: ", self.q)
        # print("omega init: ", self.omega)
        # print("omega desired init: ", self.qd)

        return self.state
    

class FaultSatellite(Satellite):
    def __init__(self, fault_mode=0, t_max=200, ts=0.1):
        super().__init__(t_max, ts)

        self.uf_buffer = []  # 单机故障力矩
        
        # 执行器故障相关
        self.fault_mode = fault_mode
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0
        self.e4 = 0

        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0

        self.uf_buffer = None

    def update_u_f(self, torque):
        self.fault_inject(self.t, self.fault_mode)
        E = np.diag([self.e1, self.e2, self.e3, self.e4])
        B = np.array([self.b1, self.b2, self.b3, self.b4])
        u_f = - E @ torque + B.reshape(-1, 1)
        # u_f要保证torque+u_f得到的值在u_max和-u_max之间
        u_f = np.clip(torque.flatten() + u_f.flatten(), -self.u_max, self.u_max).reshape(-1, 1) - torque
        self.uf_buffer.append(u_f.flatten())

    def step_fault_satellite(self, torque):
        torque = torque.reshape(-1, 1)
        self.update_u_f(torque)

    def step(self, torque):
        self.step_fault_satellite(torque)
        u = torque + self.uf_buffer[-1].reshape(-1, 1)
        return Satellite.step(self, u)

    def fault_inject(self, t, fault_mode):
        # clear
        self.e1, self.e2, self.e3, self.e4 = 0, 0, 0, 0
        self.b1, self.b2, self.b3, self.b4 = 0, 0, 0, 0
        if t < 30:
            pass
        elif t < 60:
            if fault_mode == 1:
                self.e1 = 0.4
                self.e2 = 0.3
                self.e3 = 0.6
                self.e4 = 0.2
            elif fault_mode == 2:
                self.e1 = 0.6
                self.e2 = 0.3
                self.e3 = 0.4
                self.e4 = 0.4

            if fault_mode != 0:
                self.b1 = -0.005
                self.b2 = 0
                self.b3 = 0.007
                self.b4 = 0.003
        elif t < 90:
            if fault_mode == 1:
                self.e1 = 0.4
                self.e2 = 0.5
                self.e3 = 0.4
                self.e4 = 0.1 * np.sin(0.5 * np.pi * t)
            elif fault_mode == 2:
                self.e1 = 0.7 * np.sin(0.5 * np.pi * t)
                self.e2 = 0.4
                self.e3 = 0.5 * np.sin(0.5 * np.pi * t)
                self.e4 = 0

            if fault_mode != 0:
                self.b1 = 0.002
                self.b2 = -0.003
                self.b3 = 0.005
                self.b4 = 0
        elif t < 120:
            if fault_mode == 1:
                self.e1 = 1
                self.e2 = 0.8
                self.e3 = 0.6 + 0.1 * np.cos(0.5 * np.pi * t)
                self.e4 = 0
            elif fault_mode == 2:
                self.e1 = 0.4 * np.sin(0.5 * np.pi * t)
                self.e2 = 0.2
                self.e3 = 0.3 + 0.1 * np.cos(0.5 * np.pi * t)
                self.e4 = 0.4

            if fault_mode != 0:
                self.b1 = 0
                self.b2 = 0.003
                self.b3 = 0
                self.b4 = 0.001
        else:
            if fault_mode == 1:
                self.e1 = 1
                self.e2 = 0.8
                self.e3 = 0.6 + 0.1 * np.cos(0.5 * np.pi * t)
                self.e4 = 0
            elif fault_mode == 2:
                self.e1 = 0.8
                self.e2 = 0.7
                self.e3 = 0.6
                self.e4 = 0

            if fault_mode != 0:
                self.b1 = 0
                self.b2 = 0.003
                self.b3 = 0
                self.b4 = -0.004

    def plot_fault_satellite(self):
        times = np.linspace(0, self.t_max, len(self.uf_buffer))
        uf_buffer = np.array(self.uf_buffer)

        # 绘制se
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(times, uf_buffer[:, 0], label='uf0')
        ax.plot(times, uf_buffer[:, 1], label='uf1')
        ax.plot(times, uf_buffer[:, 2], label='uf2')
        ax.plot(times, uf_buffer[:, 3], label='uf3')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('fault torque')

    def plot(self):
        self.plot_fault_satellite()
        return super().plot()

    def reset_fault_satellite(self):
        self.uf_buffer = []
        self.fault_mode = np.random.randint(0, 3)
        # print("fault mode: ", self.fault_mode)

    def reset(self):
        self.reset_fault_satellite()
        return Satellite.reset(self)


class SunPointSatellite(Satellite):
    def __init__(self, sd=np.array([[0], [0], [1]]), si=None, t_max=200, ts=0.1):
        super().__init__(t_max, ts)

        obs = np.array([5, 5, 5, 1, 1], dtype=np.float32)
        action = np.array([1, 1, 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(-action, action, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

        self.sd = sd
        self.sd = self.sd / np.linalg.norm(self.sd)
        self.si = si
        if si is None:
            self.si = np.random.random((3, 1))
        self.si = self.si / np.linalg.norm(self.si)
        self.sb = None
        self.se = None

        self.theta_buffer = []
    
    def update_se(self):
        q_correct = np.array([self.q[1], self.q[2], self.q[3], self.q[0]])
        R = Rotation.from_quat(q_correct.flatten()).as_matrix()
        self.sb = R @ self.si
        self.sb = self.sb / np.linalg.norm(self.sb)
        self.se = np.cross(self.sb.flatten(), self.sd.flatten())
        theta = np.arccos(np.dot(self.sb.flatten(), self.sd.flatten()))
        self.theta_buffer.append(theta*180/np.pi)

    def step_sun_point_satellite(self):
        self.update_se()
        omegae = self.state[4:7]
        self.state = np.concatenate([omegae.flatten(), self.se.flatten()[:2]], axis=0).flatten()

    def step(self, torque):
        torque = torque.reshape(-1, 1)
        state, _, done, info = Satellite.step(self, torque)
        self.step_sun_point_satellite()
        reward = self.reward(self.u_buffer[-1], self.omega_e_buffer[-1], self.se)
        return state, reward, done, info

    def reward(self, f, omega_e, se):
        reward_1 = 0
        reward_2 = -8 * np.linalg.norm(f)
        reward_3 = -10 * np.linalg.norm(se)
        reward_4 = -20 * np.linalg.norm(omega_e)
        reward = reward_1 + reward_2 + reward_3 + reward_4
        return reward

    def reset_sun_point_satellite(self):
        self.update_se()
        self.state = np.concatenate([self.state[4:7].flatten(), self.se.flatten()[:2]], axis=0).flatten()

    def reset(self):
        Satellite.reset(self)
        self.reset_sun_point_satellite()
        return self.state
    
    def plot_sun_point_satellite(self):
        times = np.linspace(0, self.t_max, len(self.theta_buffer))
        theta_buffer = np.array(self.theta_buffer)

        # 绘制se
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(times, theta_buffer, label='theta')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Theta')

    def plot(self):
        self.plot_sun_point_satellite()
        return super().plot()
        

class SunPointFaultSatellite(FaultSatellite, SunPointSatellite):
    def __init__(self, fault_mode=0, sd=np.array([[0], [0], [1]]), si=None, t_max=200, ts=0.1):
        FaultSatellite.__init__(self, fault_mode=fault_mode, t_max=t_max, ts=ts)
        SunPointSatellite.__init__(self, sd=sd, si=si, t_max=t_max, ts=ts)

        obs = np.array([5, 5, 5, 1, 1], dtype=np.float32)
        action = np.array([1, 1, 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(-action, action, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

    def step(self, torque):
        torque = torque.reshape(-1, 1)
        FaultSatellite.step_fault_satellite(self, torque)
        u = torque + self.uf_buffer[-1].reshape(-1, 1)
        # u = torque
        self.state, _, done, info = Satellite.step(self, u)
        SunPointSatellite.step_sun_point_satellite(self)
        reward = self.reward(self.u_buffer[-1], self.omega_e_buffer[-1], self.se)
        return self.state, reward, done, info

    def reward(self, f, omega_e, se):
        steady_state_penalty = np.exp(-np.linalg.norm(f) - np.linalg.norm(omega_e))
        reward_1 = -2 * steady_state_penalty
        reward_2 = -20 * np.linalg.norm(f)
        reward_3 = -10 * np.linalg.norm(se)
        reward_4 = -10 * np.linalg.norm(omega_e)
        reward = reward_1 + reward_2 + reward_3 + reward_4
        return reward
    
    def reset(self):
        Satellite.reset(self)
        FaultSatellite.reset_fault_satellite(self)
        SunPointSatellite.reset_sun_point_satellite(self)
        return self.state
    
    def plot(self):
        self.plot_fault_satellite()
        self.plot_sun_point_satellite()
        return Satellite.plot(self)


# test case
if __name__ == "__main__":
    env = SunPointSatellite()
    env.reset()
    for i in range(100):
        torque = np.random.random((4, 1)) * 0.1
        state, reward, done, info = env.step(torque)
        print(state)

    env.plot()


