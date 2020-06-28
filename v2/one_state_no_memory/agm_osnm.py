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
                pa = [a_i for _ in range(4)]
                qa = [a_j for _ in range(4)]
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


def build_payoff_matrix(s_i, average_payoff, id='p'):
    if id == 'p':
        apm = np.array([[average_payoff[s_i][0][0], average_payoff[s_i][0][1]],
                      [average_payoff[s_i][0][2], average_payoff[s_i][0][3]]])
    else:
        apm = np.array([[average_payoff[s_i][1][0], average_payoff[s_i][1][2]],
                      [average_payoff[s_i][1][1], average_payoff[s_i][1][3]]])
    return apm


def evolve(s_n, average_payoff, p, q, step_size, v):
    for s_i in range(s_n):
        p_m = build_payoff_matrix(s_i, average_payoff, id = 'p')
        q_m = build_payoff_matrix(s_i, average_payoff, id = 'q')
        p_o = np.dot(p_m, [[q], [1 - q]])
        q_o = np.dot(q_m, [[p], [1 - p]])
        v_s = np.sum(v)
        dp = (np.dot([1, 0], p_o)[0] - np.dot([p, 1 - p], p_o)[0]) * p * v_s
        dq = (np.dot([1, 0], q_o)[0] - np.dot([q, 1 - q], q_o)[0]) * q * v_s
        p = valid_s(p + dp * step_size)
        q = valid_s(q + dq * step_size)
    return p, q


if __name__ == '__main__':
    step_size = 0.001
    s_n = 1
    p = 0.6
    q = 0.4
    f_p = np.array([3, 1, 4, 2])
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = np.array([3, 4, 1, 2])
    f_q = f_q.reshape(f_q.size, 1).transpose()

    t = np.arange(0, 10e3)
    for _ in t:
        pl = [p, p, p, p]
        ql = [q, q, q, q]
        v, average_payoff = average_game(s_n, pl, ql, f_p, f_q)
        p, q = evolve(s_n, average_payoff, p, q, step_size, v)
        print(p, q)

