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


def calc_payoff_new(v_now, p, q, f_p, f_q, type='both'):
    m = build_markov_chain(p, q)
    # v_now = np.array([p_init * q_init, p_init * (1 - q_init), (1 - p_init) * q_init, (1 - p_init) * (1 - q_init)])
    # v_now = v_now.reshape(v_now.size, 1).transpose()
    v_next = np.dot(v_now, m).transpose()
    if type == 'p':
        r_p = np.dot(f_p, v_next)
        return r_p
    elif type == 'q':
        r_q = np.dot(f_q, v_next)
        return r_q
    else:
        r_p = np.dot(f_p, v_next)
        r_q = np.dot(f_q, v_next)
        return v_next, r_p, r_q


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
    p_0 = [0, 0, 0, 0]
    q_0 = [0, 0, 0, 0]
    p_1 = [1, 1, 1, 1]
    q_1 = [1, 1, 1, 1]
    # r_v, r_p, r_q = calc_payoff(p, q, f_p, f_q, type='both')
    # dp = (calc_payoff(sg_p, q, f_p, f_q, type='p')[1][0] - r_p[0]) * p[0]
    # dq = (calc_payoff(p, sg_q, f_p, f_q, type='q')[1][0] - r_q[0]) * q[0]
    dp = p[0] * (1 - p[0]) * (
                calc_payoff(p_1, q, f_p, f_q, type='p')[1][0] - calc_payoff(p_0, q, f_p, f_q, type='p')[1][0])
    print(calc_payoff(p_1, q, f_p, f_q, type='p')[1][0])
    dq = q[0] * (1 - q[0]) * (
                calc_payoff(p, q_1, f_p, f_q, type='q')[1][0] - calc_payoff(p, q_0, f_p, f_q, type='q')[1][0])
    for i in range(sg_n):
        p[i] = valid_s(p[i] + dp * step_size)
        q[i] = valid_s(q[i] + dq * step_size)
    return p, q


if __name__ == '__main__':
    actions = [0, 1]
    t = np.arange(0, 10e4)
    step_size = 0.001
    sg_n = 4
    # p = [0.5, 0.5, 0.5, 0.5]
    p_init = 0.4
    q_init = 0.3
    p = [p_init, p_init, p_init, p_init]
    q = [q_init, q_init, q_init, q_init]
    f_p = np.array([3, 0, 5, 1])
    f_q = np.array([3, 5, 0, 1])
    pl = []
    pl.append(p)
    ql = []
    ql.append(q)
    calc_payoff_new(p_init, q_init, p, q, f_p, f_q)
    for _ in t:
        p, q = evolve(sg_n, p, q, f_p, f_q, step_size=step_size)
        # print(p)
        # print(q)
        pl.append(p)
        ql.append(q)
