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


def build_markov_chain(qvec, p, q):
    m = np.array([[qvec[0] * p[0] * q[0], qvec[0] * p[0] * (1 - q[0]), qvec[0] * (1 - p[0]) * q[0],
                   qvec[0] * (1 - p[0]) * (1 - q[0]),
                   (1 - qvec[0]) * p[0] * q[0], (1 - qvec[0]) * p[0] * (1 - q[0]), (1 - qvec[0]) * (1 - p[0]) * q[0],
                   (1 - qvec[0]) * (1 - p[0]) * (1 - q[0])],
                  [qvec[1] * p[1] * q[1], qvec[1] * p[1] * (1 - q[1]), qvec[1] * (1 - p[1]) * q[1],
                   qvec[1] * (1 - p[1]) * (1 - q[1]),
                   (1 - qvec[1]) * p[1] * q[1], (1 - qvec[1]) * p[1] * (1 - q[1]), (1 - qvec[1]) * (1 - p[1]) * q[1],
                   (1 - qvec[1]) * (1 - p[1]) * (1 - q[1])],
                  [qvec[2] * p[2] * q[2], qvec[2] * p[2] * (1 - q[2]), qvec[2] * (1 - p[2]) * q[2],
                   qvec[2] * (1 - p[2]) * (1 - q[2]),
                   (1 - qvec[2]) * p[2] * q[2], (1 - qvec[2]) * p[2] * (1 - q[2]), (1 - qvec[2]) * (1 - p[2]) * q[2],
                   (1 - qvec[2]) * (1 - p[2]) * (1 - q[2])],
                  [qvec[3] * p[3] * q[3], qvec[3] * p[3] * (1 - q[3]), qvec[3] * (1 - p[3]) * q[3],
                   qvec[3] * (1 - p[3]) * (1 - q[3]),
                   (1 - qvec[3]) * p[3] * q[3], (1 - qvec[3]) * p[3] * (1 - q[3]), (1 - qvec[3]) * (1 - p[3]) * q[3],
                   (1 - qvec[3]) * (1 - p[3]) * (1 - q[3])],
                  [qvec[4] * p[4] * q[4], qvec[4] * p[4] * (1 - q[4]), qvec[4] * (1 - p[4]) * q[4],
                   qvec[4] * (1 - p[4]) * (1 - q[4]),
                   (1 - qvec[4]) * p[4] * q[4], (1 - qvec[4]) * p[4] * (1 - q[4]), (1 - qvec[4]) * (1 - p[4]) * q[4],
                   (1 - qvec[4]) * (1 - p[4]) * (1 - q[4])],
                  [qvec[5] * p[5] * q[5], qvec[5] * p[5] * (1 - q[5]), qvec[5] * (1 - p[5]) * q[5],
                   qvec[5] * (1 - p[5]) * (1 - q[5]),
                   (1 - qvec[5]) * p[5] * q[5], (1 - qvec[5]) * p[5] * (1 - q[5]), (1 - qvec[5]) * (1 - p[5]) * q[5],
                   (1 - qvec[5]) * (1 - p[5]) * (1 - q[5])],
                  [qvec[6] * p[6] * q[6], qvec[6] * p[6] * (1 - q[6]), qvec[6] * (1 - p[6]) * q[6],
                   qvec[6] * (1 - p[6]) * (1 - q[6]),
                   (1 - qvec[6]) * p[6] * q[6], (1 - qvec[6]) * p[6] * (1 - q[6]), (1 - qvec[6]) * (1 - p[6]) * q[6],
                   (1 - qvec[6]) * (1 - p[6]) * (1 - q[6])],
                  [qvec[7] * p[7] * q[7], qvec[7] * p[7] * (1 - q[7]), qvec[7] * (1 - p[7]) * q[7],
                   qvec[7] * (1 - p[7]) * (1 - q[7]),
                   (1 - qvec[7]) * p[7] * q[7], (1 - qvec[7]) * p[7] * (1 - q[7]), (1 - qvec[7]) * (1 - p[7]) * q[7],
                   (1 - qvec[7]) * (1 - p[7]) * (1 - q[7])]])
    return m


def determinant(m_det):
    return np.linalg.det(m_det)


def average_game_keys(a_l):
    keys_value = []
    for a_i in a_l:
        for a_j in a_l:
            keys_value.append((a_i, a_j))
    return keys_value


def calc_payoff(qvec, p, q, f_p, f_q, type='both'):
    m = build_markov_chain(qvec, p, q)
    null_matrix = np.transpose(m) - np.eye(8)
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


def evolve(sg_n, qvec, p, q, f_p, f_q, step_size):
    p_00 = [0, 0, 0, 0]
    p_10 = p[0: 4]
    q_00 = [0, 0, 0, 0]
    q_10 = q[0: 4]
    p_01 = [1, 1, 1, 1]
    p_11 = p[0: 4]
    q_01 = [1, 1, 1, 1]
    q_11 = q[0: 4]
    p_00.extend(p[4:8])
    p_10.extend([0, 0, 0, 0])
    q_00.extend(q[4:8])
    q_10.extend([0, 0, 0, 0])
    p_01.extend(p[4:8])
    p_11.extend([1, 1, 1, 1])
    q_01.extend(q[4:8])
    q_11.extend([1, 1, 1, 1])
    dp_0 = p[0] * (1 - p[0]) * (calc_payoff(qvec, p_01, q, f_p, f_q, type='p')[1][0] - calc_payoff(qvec, p_00, q, f_p, f_q, type='p')[1][0])
    dp_1 = p[4] * (1 - p[4]) * (calc_payoff(qvec, p_11, q, f_p, f_q, type='p')[1][0] - calc_payoff(qvec, p_10, q, f_p, f_q, type='p')[1][0])
    dq_0 = q[0] * (1 - q[0]) * (calc_payoff(qvec, p, q_01, f_p, f_q, type='q')[1][0] - calc_payoff(qvec, p, q_00, f_p, f_q, type='q')[1][0])
    dq_1 = q[4] * (1 - q[4]) * (calc_payoff(qvec, p, q_11, f_p, f_q, type='q')[1][0] - calc_payoff(qvec, p, q_10, f_p, f_q, type='q')[1][0])
    for i in range(4):
        p[i] = valid_s(p[i] + dp_0 * step_size)
        q[i] = valid_s(q[i] + dq_0 * step_size)
    for i in range(4, 8):
        p[i] = valid_s(p[i] + dp_1 * step_size)
        q[i] = valid_s(q[i] + dq_1 * step_size)
    return p, q

if __name__ == '__main__':
    actions = [0, 1]
    t = np.arange(0, 10e4)
    step_size = 0.001
    sg_n = 8
    # qvec = [0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9]
    qvec = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    # qvec = [0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9]
    # qvec = [0.7, 0.9, 0.3, 0.5, 0.5, 0.7, 0.1, 0.3]
    # p = [0.5, 0.5, 0.5, 0.5]
    p0 = 0.5; q0 = 0.5; p1 = 0.5; q1 = 0.5
    p = [p0, p0, p0, p0, p1, p1, p1, p1]
    q = [q0, q0, q0, q0, q1, q1, q1, q1]
    print(p, q)
    # f_p = np.array([3, 0, 5, 1, 3, 0, 5, 1])
    # f_q = np.array([3, 5, 0, 1, 3, 5, 0, 1])
    # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
    # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
    f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2])
    f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2])
    # f_p = np.array([1, -1, 2, 0, 1, -1, 2, 0])
    # f_q = np.array([1, 2, -1, 0, 1, 2, -1, 0])
    pl = []
    pl.append(p)
    ql = []
    ql.append(q)
    for _ in t:
        p, q = evolve(sg_n, qvec, p, q, f_p, f_q, step_size=step_size)
        print('p', p)
        print('q', q)
        pl.append(p)
        ql.append(q)

