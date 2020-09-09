import numpy as np
import pandas as pd
import os
import math
import datetime
from sympy import Matrix
from scipy.linalg import null_space

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


def get_payoff_pair(qvec, pl, ql, f_p, f_q):
    m = build_markov_chain(qvec, pl, ql)
    null_matrix = np.transpose(m) - np.eye(8)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    r_p = np.dot(f_p, v)[0][0]
    r_q = np.dot(f_q, v)[0][0]
    return [r_p, r_q]


if __name__ == '__main__':
    f_p = np.array([3, 1, 4, 2, 8, 6, 9, 7])
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = np.array([3, 4, 1, 2, 8, 9, 6, 7])
    f_q = f_q.reshape(f_q.size, 1).transpose()
    qvec = [12/20, 1/20, 19/20, 8/20, 12/20, 1/20, 19/20, 8/20]
    payoff_pair = []
    for _ in np.arange(10e3):
        s_r = np.random.beta(0.5, 0.5, 4)
        p0 = s_r[0]
        # q0 = s_r[1]
        q0 = 0
        p1 = s_r[2]
        # q1 = s_r[3]
        q1 = 0
        pl = [p0, p0, p0, p0, p1, p1, p1, p1]
        ql = [q0, q0, q0, q0, q1, q1, q1, q1]
        payoff_pair.append(get_payoff_pair(qvec, pl, ql, f_p, f_q))
    payoff_pair = np.array(payoff_pair)
    payoff_pair_pd = pd.DataFrame(payoff_pair)
    abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results_pp"))
    csv_file_name = "/payoff_pair_qd_ex.csv"
    file_name = abs_path + csv_file_name
    payoff_pair_pd.to_csv(file_name, index=None)

    p0 = 1
    q0 = 0
    p1 = 1
    q1 = 0
    pl = [p0, p0, p0, p0, p1, p1, p1, p1]
    ql = [q0, q0, q0, q0, q1, q1, q1, q1]
    payoff_pair_top = get_payoff_pair(qvec, pl, ql, f_p, f_q)
    payoff_pair_top = np.array(payoff_pair_top)
    payoff_pair_top_pd = pd.DataFrame(payoff_pair_top)
    csv_file_name = "/payoff_pair_qd_ex_top.csv"
    file_name = abs_path + csv_file_name
    payoff_pair_top_pd.to_csv(file_name, index=None)

    payoff_pair = []
    for value in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p0 = value
        q0 = 0
        p1 = value
        q1 = 0
        pl = [p0, p0, p0, p0, p1, p1, p1, p1]
        ql = [q0, q0, q0, q0, q1, q1, q1, q1]
        payoff_pair.append(get_payoff_pair(qvec, pl, ql, f_p, f_q))
    payoff_pair = np.array(payoff_pair)
    payoff_pair_pd = pd.DataFrame(payoff_pair)
    csv_file_name = "/payoff_pair_qd_pv_ex.csv"
    file_name = abs_path + csv_file_name
    payoff_pair_pd.to_csv(file_name, index=None)


