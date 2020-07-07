import numpy as np
import pandas as pd
import os
import math
import datetime
from sympy import Matrix
from scipy.linalg import null_space

from multiprocessing import Pool


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
                # print(v)
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
            dp = (np.dot([1, 0], p_o)[0] - np.dot([p0, 1-p0], p_o)[0]) * p0 * v_s
            dq = (np.dot([1, 0], q_o)[0] - np.dot([q0, 1-q0], q_o)[0]) * q0 * v_s
            p0 = valid_s(p0 + dp * step_size)
            q0 = valid_s(q0 + dq * step_size)
            # print(v_s)
        else:
            p_o = np.dot(p_m, [[q1], [1 - q1]])
            q_o = np.dot(q_m, [[p1], [1 - p1]])
            v_s = np.sum(v[4:8])
            dp = (np.dot([1, 0], p_o)[0] - np.dot([p1, 1-p1], p_o)[0]) * p1 * v_s
            dq = (np.dot([1, 0], q_o)[0] - np.dot([q1, 1-q1], q_o)[0]) * q1 * v_s
            p1 = valid_s(p1 + dp * step_size)
            q1 = valid_s(q1 + dq * step_size)
            # print(v_s)
    return p0, q0, p1, q1


def run_task_rd(p_init):
    t = np.arange(0, 10e5)
    step_size = 0.001
    s_n = 2
    print(p_init)
    for p_1, p_2 in [[0.9, 0.1]]:
        p0 = p_init[0]
        q0 = p_init[1]
        p1 = p_init[2]
        q1 = p_init[3]
        print(p_1, p_2)
        # qvec = [p_1, p_2, p_2, p_2, p_1, p_2, p_2, p_2]
        # qvec = [0.7, 0.9, 0.3, 0.5, 0.5, 0.7, 0.1, 0.3]
        qvec = [27/40, 6/40, 39/40, 18/40, 23/40, 2/40, 35/40, 14/40]
        f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
        # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
        f_p = f_p.reshape(f_p.size, 1).transpose()
        f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
        # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
        f_q = f_q.reshape(f_q.size, 1).transpose()
        d = []
        d.append([p0, q0, p1, q1])
        for _ in t:
            if _ % 1000 == 0:
                print('rd', _)
            pl = [p0, p0, p0, p0, p1, p1, p1, p1]
            ql = [q0, q0, q0, q0, q1, q1, q1, q1]
            v, average_payoff = average_game(s_n, qvec, pl, ql, f_p, f_q)
            p0, q0, p1, q1 = evolve(s_n, average_payoff, p0, q0, p1, q1, step_size, v)
            d.append([p0, q0, p1, q1])
        abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results"))
        csv_file_name = "/rd_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (p_init[0], p_init[1], p_init[2], p_init[3])
        file_name = abs_path + csv_file_name
        d_pd = pd.DataFrame(d)
        d_pd.to_csv(file_name, index=None)


def read_p_init():
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "p_init_file.csv")
    data = pd.read_csv(f, usecols=['0', '1', '2', '3'])
    p_init = np.array(data).tolist()
    return p_init


if __name__ == '__main__':
    p_init_list = read_p_init()
    init_num = len(p_init_list)
    p_rd = Pool()
    for _ in range(init_num):
        p_init = p_init_list[_][:]
        p_rd.apply_async(run_task_rd, args=(p_init,))
    p_rd.close()
    p_rd.join()
    print("All subprocesses done")

