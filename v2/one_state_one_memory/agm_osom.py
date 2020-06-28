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


def build_markov_chain(pl, ql):
    m = np.array([[pl[0] * ql[0], pl[0] * (1 - ql[0]), (1 - pl[0]) * ql[0], (1 - pl[0]) * (1 - ql[0])],
                  [pl[1] * ql[1], pl[1] * (1 - ql[1]), (1 - pl[1]) * ql[1], (1 - pl[1]) * (1 - ql[1])],
                  [pl[2] * ql[2], pl[2] * (1 - ql[2]), (1 - pl[2]) * ql[2], (1 - pl[2]) * (1 - ql[2])],
                  [pl[3] * ql[3], pl[3] * (1 - ql[3]), (1 - pl[3]) * ql[3], (1 - pl[3]) * (1 - ql[3])]])
    return m


def average_game(s_n, pl, ql, f_p, f_q):
    average_payoff = []
    for i in range(s_n):
        average_payoff.append([[0 for _ in range(4)], [0 for _ in range(4)]])
    for s_i in range(s_n):
        for a_i in [1, 0]:
            for a_j in [1, 0]:
                pa = pl[:]
                qa = ql[:]
                pa[s_i] = a_i
                qa[s_i] = a_j
                m = build_markov_chain(pa, qa)
                null_matrix = np.transpose(m) - np.eye(4)
                v = null_space(null_matrix)
                v = v / np.sum(v)
                r_p = np.dot(f_p, v)[0][0]
                r_q = np.dot(f_q, v)[0][0]
                average_payoff[s_i][0][(1 - a_i) * 2 + (1 - a_j)] = r_p
                average_payoff[s_i][1][(1 - a_i) * 2 + (1 - a_j)] = r_q
    m = build_markov_chain(pl, ql)
    null_matrix = np.transpose(m) - np.eye(4)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    v = np.array(v)
    v = v.transpose()[0]
    return v, average_payoff


def build_payoff_matrix(s_i, average_payoff, id = 'p'):
    if id == 'p':
        apm = np.array([[average_payoff[s_i][0][0], average_payoff[s_i][0][1]],
                       [average_payoff[s_i][0][2], average_payoff[s_i][0][3]]])
    else:
        apm = np.array([[average_payoff[s_i][1][0], average_payoff[s_i][1][2]],
                       [average_payoff[s_i][1][1], average_payoff[s_i][1][3]]])
    return apm


def evolve(s_n, average_payoff, pl, ql, step_size, v):
    pl_t = pl[:]
    ql_t = ql[:]
    pl_r = [0 for _ in range(4)]
    ql_r = [0 for _ in range(4)]
    for s_i in range(s_n):
        p_m = build_payoff_matrix(s_i, average_payoff, id='p')
        # print('p_m', p_m)
        q_m = build_payoff_matrix(s_i, average_payoff, id='q')
        # print('q_m', q_m)
        p_o = np.dot(p_m, [[ql_t[s_i]], [1 - ql_t[s_i]]])
        q_o = np.dot(q_m, [[pl_t[s_i]], [1 - pl_t[s_i]]])
        # dp = (np.dot([1, 0], p_o)[0] - np.dot([pl_t[s_i], 1 - pl_t[s_i]], p_o)[0]) * pl_t[s_i] * v[s_i]
        dp = (np.dot([1, 0], p_o)[0] - np.dot([0, 1], p_o)[0]) * pl_t[s_i] * (1 - pl_t[s_i]) * v[s_i]
        # dq = (np.dot([1, 0], q_o)[0] - np.dot([ql_t[s_i], 1 - ql_t[s_i]], q_o)[0]) * ql_t[s_i] * v[s_i]
        dq = (np.dot([1, 0], q_o)[0] - np.dot([0, 1], q_o)[0]) * ql_t[s_i] * (1 - ql_t[s_i]) * v[s_i]
        pl_r[s_i] = valid_s(pl_t[s_i] + dp * step_size)
        ql_r[s_i] = valid_s(ql_t[s_i] + dq * step_size)
    return ql_r
    # return pl_r, ql_r


if __name__ == '__main__':
    step_size = 0.001
    s_n = 4
    # pl = [0.5, 0.5, 0.5, 0.5]
    pl = [11/13, 1/2, 7/26, 0.01]
    ql = [0.5, 0.5, 0.5, 0.5]
    f_p = np.array([3, 0, 5, 1])
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = np.array([3, 5, 0, 1])
    f_q = f_q.reshape(f_q.size, 1).transpose()

    t = np.arange(0, 10e4)
    for _ in t:
        v, average_payoff = average_game(s_n, pl, ql, f_p, f_q)
        ql = evolve(s_n, average_payoff, pl, ql, step_size, v)
        print(pl, ql)
