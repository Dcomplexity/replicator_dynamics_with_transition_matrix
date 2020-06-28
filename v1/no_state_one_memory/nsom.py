import numpy as np
import pandas as pd
import os
import math
import datetime
from sympy import Matrix
from scipy.linalg import null_space


def valid_s(s_value):
    if s_value < 0.0:
        s_new = 0.0
    elif s_value > 1.0:
        s_new = 1.0
    else:
        s_new = s_value
    return s_new


def build_markov_chain(p, q):
    m = np.array([[p[0] * q[0], p[0] * (1 - q[0]), (1 - p[0]) * q[0], (1 - p[0]) * (1 - q[0])],
                  [p[1] * q[1], p[1] * (1 - q[1]), (1 - p[1]) * q[1], (1 - p[1]) * (1 - q[1])],
                  [p[2] * q[2], p[2] * (1 - q[2]), (1 - p[2]) * q[2], (1 - p[2]) * (1 - q[2])],
                  [p[3] * q[3], p[3] * (1 - q[3]), (1 - p[3]) * q[3], (1 - p[3]) * (1 - q[3])]])
    return m


def determinant(m_det):
    return np.linalg.det(m_det)


def average_game_keys(a_l):
    keys_value = []
    for a_i in a_l:
        for a_j in a_l:
            keys_value.append((a_i, a_j))
    return keys_value


def calc_payoff(p, q, f_p, f_q, type='both'):
    m = build_markov_chain(p, q)
    null_matrix = np.transpose(m) - np.eye(4)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    if type == 'p':
        f_p = f_p.reshape(f_p.size, 1).transpose()
        r_p = np.dot(f_p, v)[0]
        v = v.flatten()
        return v, r_p
    if type == 'q':
        f_q = f_q.reshape(f_q.size, 1).transpose()
        r_q = np.dot(f_q, v)[0]
        v = v.flatten()
        return v, r_q
    if type == 'both':
        f_p = f_p.reshape(f_p.size, 1).transpose()
        f_q = f_q.reshape(f_q.size, 1).transpose()
        r_p = np.dot(f_p, v)[0]
        r_q = np.dot(f_q, v)[0]
        v = v.flatten()
        return v, r_p, r_q
    else:
        return 0


# def average_payoff_matrix(p, q, f_p, f_q, keys_value):
#     r_dict = {}
#     for i in range(len(keys_value)):
#         r_dict[keys_value[i]] = [0.0, 0.0]
#     for key_item in keys_value:
#         a_p = key_item[0][0]
#         a_q = key_item[0][1]
#         v, r_p, r_q = calc_payoff(p, q, f_p, f_q)
#         r_dict[key_item][0] = r_p[0]
#         r_dict[key_item][1] = r_p[1]
#     return r_dict


def evolve(sg_n, p, q, f_p, f_q, step_size):
    p_0 = [p[:] for i in range(4)]
    q_0 = [q[:] for i in range(4)]
    p_1 = [p[:] for i in range(4)]
    q_1 = [q[:] for i in range(4)]
    for i in range(sg_n):
        p_0[i][i] = 0.0
        q_0[i][i] = 0.0
        p_1[i][i] = 1.0
        q_1[i][i] = 1.0
    # r_v, r_p, r_q = calc_payoff(p, q, f_p, f_q, type='both')
    dp = []
    dq = []
    for i in range(sg_n):
        # dp.append((calc_payoff(p_1[i], q, f_p, f_q, type='p')[1][0] - p[i] *
        #            calc_payoff(p_1[i], q, f_p, f_q, type='p')[1][0] - (1 - p[i]) *
        #            calc_payoff(p_0[i], q, f_p, f_q, type='p')[1][0]) * p[i])
        # dq.append((calc_payoff(p, q_1[i], f_p, f_q, type='q')[1][0] - q[i] *
        #            calc_payoff(p, q_1[i], f_p, f_q, type='q')[1][0] - (1 - q[i]) *
        #            calc_payoff(p, q_0[i], f_p, f_q, type='q')[1][0]) * q[i])
        dp.append(p[i] * (1 - p[i]) * (
                calc_payoff(p_1[i], q, f_p, f_q, type='p')[1][0] - calc_payoff(p_0[i], q, f_p, f_q, type='p')[1][0]))
        dq.append(q[i] * (1 - q[i]) * (
                calc_payoff(p, q_1[i], f_p, f_q, type='q')[1][0] - calc_payoff(p, q_0[i], f_p, f_q, type='q')[1][0]))
        # if i == 0:
        #     print(calc_payoff(p_1[i], q, f_p, f_q, type='p')[1][0], calc_payoff(p_0[i], q, f_p, f_q, type='p')[1][0])
        #     print(calc_payoff(p, q_1[i], f_p, f_q, type='q')[1][0], calc_payoff(p, q_0[i], f_p, f_q, type='q')[1][0])
        # if i == 0:
        #     print(calc_payoff(p_1[i], q, f_p, f_q, type='p')[0], calc_payoff(p_0[i], q, f_p, f_q, type='p')[0])
        #     print(calc_payoff(p, q_1[i], f_p, f_q, type='q')[0], calc_payoff(p, q_0[i], f_p, f_q, type='q')[0])
        #     print(p, q)
    # print(dp)
    # print(dq)
    for i in range(sg_n):
        p[i] = valid_s(p[i] + dp[i] * step_size)
        q[i] = valid_s(q[i] + dq[i] * step_size)
    return p, q


if __name__ == '__main__':
    actions = [0, 1]
    t = np.arange(0, 10e5)
    step_size = 0.001
    sg_n = 4
    # p = [0.5, 0.5, 0.5, 0.5]
    p_init = 0.9
    q_init = 0.9
    p = [p_init, p_init, p_init, p_init]
    # p = [11 / 13, 1 / 2, 7 / 26, 0]
    q = [q_init, q_init, q_init, q_init]
    f_p = np.array([3, 0, 5, 1])
    f_q = np.array([3, 5, 0, 1])
    pl = []
    pl.append(p)
    ql = []
    ql.append(q)
    for _ in t:
        p, q = evolve(sg_n, p, q, f_p, f_q, step_size=step_size)
        print(p)
        print(q)
        pl.append(p)
        ql.append(q)
