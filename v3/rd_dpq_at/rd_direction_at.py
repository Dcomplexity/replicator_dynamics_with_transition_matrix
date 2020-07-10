import numpy as np
import pandas as pd
import os
import math
import datetime
from sympy import Matrix
from scipy.linalg import null_space
import itertools
from multiprocessing import Pool


def build_markov_chain(qvec, p, q):
    m = np.array([[qvec[0] * p[0] * q[0], qvec[0] * p[0] * (1 - q[0]), qvec[0] * (1 - p[0]) * q[0],
                   qvec[0] * (1 - p[0]) * (1 - q[0]),
                   (1 - qvec[0]) * p[4] * q[4], (1 - qvec[0]) * p[4] * (1 - q[4]), (1 - qvec[0]) * (1 - p[4]) * q[4],
                   (1 - qvec[0]) * (1 - p[4]) * (1 - q[4])],
                  [qvec[1] * p[1] * q[1], qvec[1] * p[1] * (1 - q[1]), qvec[1] * (1 - p[1]) * q[1],
                   qvec[1] * (1 - p[1]) * (1 - q[1]),
                   (1 - qvec[1]) * p[5] * q[5], (1 - qvec[1]) * p[5] * (1 - q[5]), (1 - qvec[1]) * (1 - p[5]) * q[5],
                   (1 - qvec[1]) * (1 - p[5]) * (1 - q[5])],
                  [qvec[2] * p[2] * q[2], qvec[2] * p[2] * (1 - q[2]), qvec[2] * (1 - p[2]) * q[2],
                   qvec[2] * (1 - p[2]) * (1 - q[2]),
                   (1 - qvec[2]) * p[6] * q[6], (1 - qvec[2]) * p[6] * (1 - q[6]), (1 - qvec[2]) * (1 - p[6]) * q[6],
                   (1 - qvec[2]) * (1 - p[6]) * (1 - q[6])],
                  [qvec[3] * p[3] * q[3], qvec[3] * p[3] * (1 - q[3]), qvec[3] * (1 - p[3]) * q[3],
                   qvec[3] * (1 - p[3]) * (1 - q[3]),
                   (1 - qvec[3]) * p[7] * q[7], (1 - qvec[3]) * p[7] * (1 - q[7]), (1 - qvec[3]) * (1 - p[7]) * q[7],
                   (1 - qvec[3]) * (1 - p[7]) * (1 - q[7])],
                  [qvec[4] * p[0] * q[0], qvec[4] * p[0] * (1 - q[0]), qvec[4] * (1 - p[0]) * q[0],
                   qvec[4] * (1 - p[0]) * (1 - q[0]),
                   (1 - qvec[4]) * p[4] * q[4], (1 - qvec[4]) * p[4] * (1 - q[4]), (1 - qvec[4]) * (1 - p[4]) * q[4],
                   (1 - qvec[4]) * (1 - p[4]) * (1 - q[4])],
                  [qvec[5] * p[1] * q[1], qvec[5] * p[1] * (1 - q[1]), qvec[5] * (1 - p[1]) * q[1],
                   qvec[5] * (1 - p[1]) * (1 - q[1]),
                   (1 - qvec[5]) * p[5] * q[5], (1 - qvec[5]) * p[5] * (1 - q[5]), (1 - qvec[5]) * (1 - p[5]) * q[5],
                   (1 - qvec[5]) * (1 - p[5]) * (1 - q[5])],
                  [qvec[6] * p[2] * q[2], qvec[6] * p[2] * (1 - q[2]), qvec[6] * (1 - p[2]) * q[2],
                   qvec[6] * (1 - p[2]) * (1 - q[2]),
                   (1 - qvec[6]) * p[6] * q[6], (1 - qvec[6]) * p[6] * (1 - q[6]), (1 - qvec[6]) * (1 - p[6]) * q[6],
                   (1 - qvec[6]) * (1 - p[6]) * (1 - q[6])],
                  [qvec[7] * p[3] * q[3], qvec[7] * p[3] * (1 - q[3]), qvec[7] * (1 - p[3]) * q[3],
                   qvec[7] * (1 - p[3]) * (1 - q[3]),
                   (1 - qvec[7]) * p[7] * q[7], (1 - qvec[7]) * p[7] * (1 - q[7]), (1 - qvec[7]) * (1 - p[7]) * q[7],
                   (1 - qvec[7]) * (1 - p[7]) * (1 - q[7])]])
    return m


def average_game(s_n, qvec, pl, ql, f_p, f_q):
    average_payoff = []
    for i in range(s_n):
        average_payoff.append([[0 for _ in range(4)], [0 for _ in range(4)]])
    for s_i in range(s_n):
        for a_i in [1, 0]:
            for a_j in [1, 0]:
                pa = pl[:]
                qa = ql[:]
                for s_j in range(4):
                    pa[s_i * 4 + s_j] = a_i
                    qa[s_i * 4 + s_j] = a_j
                m = build_markov_chain(qvec, pa, qa)
                null_matrix = np.transpose(m) - np.eye(8)
                v = null_space(null_matrix)
                v = v / np.sum(v)
                r_p = np.dot(f_p, v)[0][0]
                r_q = np.dot(f_q, v)[0][0]
                average_payoff[s_i][0][(1 - a_i) * 2 + (1 - a_j)] = r_p
                average_payoff[s_i][1][(1 - a_i) * 2 + (1 - a_j)] = r_q
    m = build_markov_chain(qvec, pl, ql)
    null_matrix = np.transpose(m) - np.eye(8)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    v = np.array(v)
    v = v.transpose()[0]
    return v, average_payoff


def build_payoff_matrix(s_i, average_payoff, id ='p'):
    if id == 'p':
        apm = np.array([[average_payoff[s_i][0][0], average_payoff[s_i][0][1]],
                      [average_payoff[s_i][0][2], average_payoff[s_i][0][3]]])
    else:
        apm = np.array([[average_payoff[s_i][1][0], average_payoff[s_i][1][2]],
                      [average_payoff[s_i][1][1], average_payoff[s_i][1][3]]])
    return apm


def evolve(s_n, average_payoff, p0, q0, p1, q1, step_size, v):
    for s_i in range(s_n):
        p_m = build_payoff_matrix(s_i, average_payoff, id = 'p')
        # print('p_m', s_i, p_m)
        q_m = build_payoff_matrix(s_i, average_payoff, id = 'q')
        # print('q_m', s_i, q_m)
        if s_i == 0:
            p_o = np.dot(p_m, [[q0], [1 - q0]])
            q_o = np.dot(q_m, [[p0], [1 - p0]])
            v_s = np.sum(v[0:4])
            dp0 = (np.dot([1, 0], p_o)[0] - np.dot([p0, 1-p0], p_o)[0]) * p0 * v_s
            dq0 = (np.dot([1, 0], q_o)[0] - np.dot([q0, 1-q0], q_o)[0]) * q0 * v_s
        else:
            p_o = np.dot(p_m, [[q1], [1 - q1]])
            q_o = np.dot(q_m, [[p1], [1 - p1]])
            v_s = np.sum(v[4:8])
            dp1 = (np.dot([1, 0], p_o)[0] - np.dot([p1, 1-p1], p_o)[0]) * p1 * v_s
            dq1 = (np.dot([1, 0], q_o)[0] - np.dot([q1, 1-q1], q_o)[0]) * q1 * v_s
    return dp0, dq0, dp1, dq1


def run_task():
    t = np.arange(0, 10e4)
    step_size = 0.01
    s_n = 2
    for p_1, p_2 in [[0.9, 0.1], [0.5, 0.5]]:
        print(p_1, p_2)
        p_sub = np.round(np.arange(0.1, 1.0, 0.1), 2)
        p_prod_s1 = list(itertools.product(p_sub, p_sub))
        p_prod = np.zeros((len(p_prod_s1), 4))
        for i in range(len(p_prod_s1)):
            p_prod[i][0] = p_prod_s1[i][0]
            p_prod[i][1] = p_prod_s1[i][1]
        qvec = [p_1, p_2, p_2, p_2, p_1, p_2, p_2, p_2]
        f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2])
        # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
        f_p = f_p.reshape(f_p.size, 1).transpose()
        f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2])
        # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
        f_q = f_q.reshape(f_q.size, 1).transpose()
        d = []
        for p_init in p_prod:
            print(p_init)
            p0 = p_init[0]
            q0 = p_init[1]
            p1 = p_init[2]
            q1 = p_init[3]
            pl = [p0, p0, p0, p0, p1, p1, p1, p1]
            ql = [q0, q0, q0, q0, q1, q1, q1, q1]
            v, average_payoff = average_game(s_n, qvec, pl, ql, f_p, f_q)
            dp0, dq0, dp1, dq1 = evolve(s_n, average_payoff, p0, q0, p1, q1, step_size, v)
            d.append([dp0, dq0, dp1, dq1])
        abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results_dpq_at"))
        csv_file_name = "/direction_p1_%.1f_p2_%.1f.csv" % (p_1, p_2)
        file_name = abs_path + csv_file_name
        d_pd = pd.DataFrame(d)
        d_pd.to_csv(file_name, index=None)


if __name__ == '__main__':
    run_task()