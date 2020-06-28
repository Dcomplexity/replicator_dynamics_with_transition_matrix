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


def calc_payoff_new(p, q, f_p, f_q, type='both'):
    pl = [p for _ in range(4)]
    ql = [q for _ in range(4)]
    m = build_markov_chain(pl, ql)
    v_now = np.array([p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)])
    v_now = v_now.reshape(v_now.size, 1).transpose()
    v_next = np.dot(v_now, m).transpose()
    if type == 'p':
        r_p = np.dot(f_p, v_next)[0]
        return r_p
    elif type == 'q':
        r_q = np.dot(f_q, v_next)[0]
        return r_q
    else:
        return 0


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


def evolve(p, q, f_p, f_q, step_size):
    dp = p * (1 - p) * (
                calc_payoff_new(1, q, f_p, f_q, type='p') - calc_payoff_new(0, q, f_p, f_q, type='p'))
    dq = q * (1 - q) * (
                calc_payoff_new(p, 1, f_p, f_q, type='q') - calc_payoff_new(p, 0, f_p, f_q, type='q'))
    p = valid_s(p + dp * step_size)
    q = valid_s(q + dq * step_size)
    return p, q


if __name__ == '__main__':
    actions = [0, 1]
    t = np.arange(0, 10e3)
    step_size = 0.001
    p = 0.5
    q = 0.5
    # p = [p_init, p_init, p_init, p_init]
    f_p = np.array([3, 0, 5, 1])
    f_q = np.array([3, 5, 0, 1])
    pl = []
    pl.append(p)
    ql = []
    ql.append(q)
    for _ in t:
        p, q = evolve(p, q, f_p, f_q, step_size=step_size)
        print(p, q)
        pl.append(p)
        ql.append(q)
