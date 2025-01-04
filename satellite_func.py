import numpy as np


def q_x(a):  # 叉乘矩阵
    a_x = np.array([
        [0, -float(a[2]), float(a[1])],
        [float(a[2]), 0, -float(a[0])],
        [-float(a[1]), float(a[0]), 0]
    ])
    return a_x


def d_q(a, b):  # 四元数的导数
    qv = np.array([
        [float(a[1][0])],
        [float(a[2][0])],
        [float(a[3][0])]
    ])
    qr = float(a[0][0])
    B = q_x(qv) + qr * np.eye(3)
    qv = qv.T
    C = np.block([
        [0, -qv],
        [np.zeros((3, 1)), B]
    ])
    D = np.array([
        [0],
        [float(b[0][0])],
        [float(b[1][0])],
        [float(b[2][0])]
    ])
    return 0.5 * C @ D


def d_omega(j_1, omega, j, u):  # 角加速度
    D_omega = j_1 @ ((-q_x(omega) @ j @ omega) + u)
    return D_omega


def get_q_e(q_d, q):  # 误差四元数
    q_dv = np.array([
        [float(q_d[1][0])],
        [float(q_d[2][0])],
        [float(q_d[3][0])]
    ])
    q_d0 = float(q_d[0][0])
    matiq_d = np.block([
        [q_d.T],
        [-q_dv, q_d0 * np.eye(3) - q_x(q_dv)]
    ])
    qe = matiq_d @ q
    return qe


def get_omega_e(omega, omega_d, qe):  # 误差角速度
    qe0 = float(qe[0][0])
    qev = np.array([
        [float(qe[1][0])],
        [float(qe[2][0])],
        [float(qe[3][0])]
    ])
    C_i = (qe0 ** 2 - (qev.T @ qev)) * np.eye(3) + 2 * qev @ qev.T - 2 * qe0 * q_x(qev)
    omega_e1 = omega - C_i @ omega_d
    return omega_e1


def q_self(q_d, q_e):
    q_dv = np.array([
        [float(q_d[1][0])],
        [float(q_d[2][0])],
        [float(q_d[3][0])]
    ])
    q_d0 = float(q_d[0][0])
    matiq_d = np.block([
        [q_d.T],
        [q_dv, q_d0 * np.eye(3) - q_x(q_dv)]
    ])
    print(matiq_d)
    q = np.linalg.inv(matiq_d) @ q_e
    return q


def R_K(q, omega, tau, j_1, j, u):
    K_21 = d_omega(j_1, omega, j, u)
    K_22 = d_omega(j_1, omega + K_21 / 2 * tau, j, u)
    K_23 = d_omega(j_1, omega + K_22 / 2 * tau, j, u)
    K_24 = d_omega(j_1, omega + K_23 * tau, j, u)
    omega = omega + (K_21 + 2 * K_22 + 2 * K_23 + K_24) / 6 * tau

    K_11 = d_q(q, omega)
    K_12 = d_q(q + K_11 / 2 * tau, omega)
    K_13 = d_q(q + K_12 / 2 * tau, omega)
    K_14 = d_q(q + K_13 * tau, omega)
    q = q + (K_11 + 2 * K_12 + 2 * K_13 + K_14) / 6 * tau

    return q, omega


def Rk_q(q, omega, tau):
    K_11 = d_q(q, omega)
    K_12 = d_q(q + K_11 / 2 * tau, omega)
    K_13 = d_q(q + K_12 / 2 * tau, omega)
    K_14 = d_q(q + K_13 * tau, omega)
    q = q + (K_11 + 2 * K_12 + 2 * K_13 + K_14) / 6 * tau
    return q


def get_omega_d(t):
    omega_d = np.array([
        [0],
        [0],
        [0]
    ])
    return omega_d
