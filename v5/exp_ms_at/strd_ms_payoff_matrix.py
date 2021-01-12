import numpy as np
import pandas as pd
import os
import math
import datetime
from sympy import Matrix
from scipy.linalg import null_space

from multiprocessing import Pool


# def valid_s(s_value):
#     if s_value < 0.0:
#         s_new = 0.0
#     elif s_value > 1.0:
#         s_new = 1.0
#     else:
#         s_new = s_value
#     return s_new


def build_markov_chain(qvec, p, q):
    m = np.ones((12, 12))
    for i in range(np.size(m, axis=0)):
        for j in range(np.size(m, axis=1)):
            if j % 4 == 0:
                m[i][j] = qvec[i][j // 4] * p[j // 4] * q[j // 4]
            elif j % 4 == 1:
                m[i][j] = qvec[i][j // 4] * p[j // 4] * (1 - q[j // 4])
            elif j % 4 == 2:
                m[i][j] = qvec[i][j // 4] * (1 - p[j // 4]) * q[j // 4]
            else:
                m[i][j] = qvec[i][j // 4] * (1 - p[j // 4]) * (1 - q[j // 4])
    print(m)
    return m


def calc_expected_payoff(qvec, pl, ql, f_p, f_q):
    pa = pl[:]
    qa = ql[:]
    m = build_markov_chain(qvec, pa, qa)
    # print(m)
    null_matrix = np.transpose(m) - np.eye(12)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    # print(v)
    r_p_e = np.dot(f_p, v)[0][0]
    r_q_e = np.dot(f_q, v)[0][0]
    return v, r_p_e, r_q_e


def calc_payoff(agent_id, s, a_p, a_q, qvec, pl, ql, f_p, f_q):
    action_payoff = 0
    if agent_id == 0:
        pa = pl[:]
        qa = ql[:]
        pa[s] = a_p
        qa[s] = a_q
        m = build_markov_chain(qvec, pa, qa)
        null_matrix = np.transpose(m) - np.eye(12)
        v = null_space(null_matrix)
        v = v / np.sum(v)
        action_payoff = np.dot(f_p, v)[0][0]
        return action_payoff
    elif agent_id == 1:
        pa = pl[:]
        qa = ql[:]
        pa[s] = a_p
        qa[s] = a_q
        m = build_markov_chain(qvec, pa, qa)
        null_matrix = np.transpose(m) - np.eye(12)
        v = null_space(null_matrix)
        v = v / np.sum(v)
        action_payoff = np.dot(f_q, v)[0][0]
        return action_payoff


# def run_task_rd(s_init):
#     t = np.arange(0, int(10e5))
#     step_size = 0.001
#     s_n = 3
#     print(s_init)
#     for z_1, z_2 in [[0.9, 0.1]]:
#         p0 = s_init[0]
#         q0 = s_init[1]
#         p1 = s_init[2]
#         q1 = s_init[3]
#         p2 = s_init[4]
#         q2 = s_init[5]
#         print(z_1, z_2)
#         # qvec = [z_1, z_2, z_2, z_2, z_1, z_2, z_2, z_2]
#         qmatrix = [[0.8, 0.1, 0.1], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45],
#                    [0.8, 0.1, 0.1], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45],
#                    [0.8, 0.1, 0.1], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45]]
#         qmatrix0 = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
#                     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
#                     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
#         qmatrix1 = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
#                     [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
#                     [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
#         qmatrix2 = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
#                     [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
#                     [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
#         f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2])
#         # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
#         f_p = f_p.reshape(f_p.size, 1).transpose()
#         f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2])
#         # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
#         f_q = f_q.reshape(f_q.size, 1).transpose()
#         d = []
#         d.append([p0, q0, p1, q1, p2, q2])
#         for _ in t:
#             # print('strd', _)
#             if _ % 1000 == 0:
#                 print('strd', _)
#             pl = [p0, p1, p2]
#             ql = [q0, q1, q2]
#             # pl = [p0, p0, p0, p0, p1, p1, p1, p1]
#             # ql = [q0, q0, q0, q0, q1, q1, q1, q1]
#             # calc_expected_payoff(qvec, pl, ql, f_p, f_q)
#             v, r_p_e, r_q_e = calc_expected_payoff(qmatrix, pl, ql, f_p, f_q)
#             # calc_payoff(agent_id, s, a_p, a_q, qvec, pl, ql, f_p, f_q)
#             r_p_0_cc = calc_payoff(0, 0, 1, 1, qmatrix, pl, ql, f_p, f_q)
#             r_q_0_cc = calc_payoff(1, 0, 1, 1, qmatrix, pl, ql, f_p, f_q)
#             r_p_0_cd = calc_payoff(0, 0, 1, 0, qmatrix, pl, ql, f_p, f_q)
#             r_q_0_cd = calc_payoff(1, 0, 1, 0, qmatrix, pl, ql, f_p, f_q)
#             r_p_0_dc = calc_payoff(0, 0, 0, 1, qmatrix, pl, ql, f_p, f_q)
#             r_q_0_dc = calc_payoff(1, 0, 0, 1, qmatrix, pl, ql, f_p, f_q)
#             r_p_0_dd = calc_payoff(0, 0, 0, 0, qmatrix, pl, ql, f_p, f_q)
#             r_q_0_dd = calc_payoff(1, 0, 0, 0, qmatrix, pl, ql, f_p, f_q)
#             r_p_1_cc = calc_payoff(0, 1, 1, 1, qmatrix, pl, ql, f_p, f_q)
#             r_q_1_cc = calc_payoff(1, 1, 1, 1, qmatrix, pl, ql, f_p, f_q)
#             r_p_1_cd = calc_payoff(0, 1, 1, 0, qmatrix, pl, ql, f_p, f_q)
#             r_q_1_cd = calc_payoff(1, 1, 1, 0, qmatrix, pl, ql, f_p, f_q)
#             r_p_1_dc = calc_payoff(0, 1, 0, 1, qmatrix, pl, ql, f_p, f_q)
#             r_q_1_dc = calc_payoff(1, 1, 0, 1, qmatrix, pl, ql, f_p, f_q)
#             r_p_1_dd = calc_payoff(0, 1, 0, 0, qmatrix, pl, ql, f_p, f_q)
#             r_q_1_dd = calc_payoff(1, 1, 0, 0, qmatrix, pl, ql, f_p, f_q)
#             r_p_2_cc = calc_payoff(0, 2, 1, 1, qmatrix, pl, ql, f_p, f_q)
#             r_q_2_cc = calc_payoff(1, 2, 1, 1, qmatrix, pl, ql, f_p, f_q)
#             r_p_2_cd = calc_payoff(0, 2, 1, 0, qmatrix, pl, ql, f_p, f_q)
#             r_q_2_cd = calc_payoff(1, 2, 1, 0, qmatrix, pl, ql, f_p, f_q)
#             r_p_2_dc = calc_payoff(0, 2, 0, 1, qmatrix, pl, ql, f_p, f_q)
#             r_q_2_dc = calc_payoff(1, 2, 0, 1, qmatrix, pl, ql, f_p, f_q)
#             r_p_2_dd = calc_payoff(0, 2, 0, 0, qmatrix, pl, ql, f_p, f_q)
#             r_q_2_dd = calc_payoff(1, 2, 0, 0, qmatrix, pl, ql, f_p, f_q)
#             v_0 = np.sum(v[0:4])
#             v_1 = np.sum(v[4:8])
#             v_2 = np.sum(v[8:12])
#             f_p_exp = np.array(
#                 [r_p_0_cc, r_p_0_cd, r_p_0_dc, r_p_0_dd, r_p_1_cc, r_p_1_cd, r_p_1_dc, r_p_1_dd, r_p_2_cc, r_p_2_cd,
#                  r_p_2_dc, r_p_2_dd])
#             f_p_exp = f_p_exp.reshape(f_p_exp.size, 1).transpose()
#             f_q_exp = np.array(
#                 [r_q_0_cc, r_q_0_cd, r_q_0_dc, r_q_0_dd, r_q_1_cc, r_q_1_cd, r_q_1_dc, r_q_1_dd, r_q_2_cc, r_q_2_cd,
#                  r_q_2_dc, r_q_2_dd])
#             f_q_exp = f_q_exp.reshape(f_q_exp.size, 1).transpose()
#             # dp0 = ((r_p_0_cc * q0 + r_p_0_cd * (1 - q0)) - (p0 * (r_p_0_cc * q0 + r_p_0_cd * (1 - q0)) + (1 - p0) * (
#             #         r_p_0_dc * q0 + r_p_0_dd * (1 - q0)))) * p0 * v_0
#             # dq0 = ((r_q_0_cc * p0 + r_q_0_dc * (1 - p0)) - (q0 * (r_q_0_cc * p0 + r_q_0_dc * (1 - p0)) + (1 - q0) * (
#             #         r_q_0_cd * p0 + r_q_0_dd * (1 - p0)))) * q0 * v_0
#             # dp1 = ((r_p_1_cc * q1 + r_p_1_cd * (1 - q1)) - (p1 * (r_p_1_cc * q1 + r_p_1_cd * (1 - q1)) + (1 - p1) * (
#             #         r_p_1_dc * q1 + r_p_1_dd * (1 - q1)))) * p1 * v_1
#             # dq1 = ((r_q_1_cc * p1 + r_q_1_dc * (1 - p1)) - (q1 * (r_q_1_cc * p1 + r_q_1_dc * (1 - p1)) + (1 - q1) * (
#             #         r_q_1_cd * p1 + r_q_1_dd * (1 - p1)))) * q1 * v_1
#             dp0 = (calc_payoff(0, 0, 1, q0, qmatrix0, pl, ql, f_p_exp, f_q_exp) - calc_payoff(0, 0, p0, q0, qmatrix0,
#                                                                                               pl, ql,
#                                                                                               f_p_exp,
#                                                                                               f_q_exp)) * p0 * v_0
#             dq0 = (calc_payoff(1, 0, p0, 1, qmatrix0, pl, ql, f_p_exp, f_q_exp) - calc_payoff(1, 0, p0, q0, qmatrix0,
#                                                                                               pl, ql,
#                                                                                               f_p_exp,
#                                                                                               f_q_exp)) * q0 * v_0
#             dp1 = (calc_payoff(0, 1, 1, q1, qmatrix1, pl, ql, f_p_exp, f_q_exp) - calc_payoff(0, 1, p1, q1, qmatrix1,
#                                                                                               pl, ql,
#                                                                                               f_p_exp,
#                                                                                               f_q_exp)) * p1 * v_1
#             dq1 = (calc_payoff(1, 1, p1, 1, qmatrix1, pl, ql, f_p_exp, f_q_exp) - calc_payoff(1, 1, p1, q1, qmatrix1,
#                                                                                               pl, ql,
#                                                                                               f_p_exp,
#                                                                                               f_q_exp)) * q1 * v_1
#             dp2 = (calc_payoff(0, 2, 1, q2, qmatrix2, pl, ql, f_p_exp, f_q_exp) - calc_payoff(0, 2, p2, q2, qmatrix2,
#                                                                                               pl, ql,
#                                                                                               f_p_exp,
#                                                                                               f_q_exp)) * p2 * v_2
#             dq2 = (calc_payoff(1, 2, p2, 1, qmatrix2, pl, ql, f_p_exp, f_q_exp) - calc_payoff(1, 2, p2, q2, qmatrix2,
#                                                                                               pl, ql,
#                                                                                               f_p_exp,
#                                                                                               f_q_exp)) * q2 * v_2
#             p0 = p0 + dp0 * step_size
#             q0 = q0 + dq0 * step_size
#             p1 = p1 + dp1 * step_size
#             q1 = q1 + dq1 * step_size
#             p2 = p2 + dp2 * step_size
#             q2 = q2 + dq2 * step_size
#             d.append([p0, q0, p1, q1, p2, q2])
#         abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results_st_at"))
#         csv_file_name = "/strd_ms_st_at_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (
#             s_init[0], s_init[1], s_init[2], s_init[3], s_init[4], s_init[5])
#         file_name = abs_path + csv_file_name
#         d_pd = pd.DataFrame(d)
#         d_pd.to_csv(file_name, index=None)
#
#
# def read_s_init():
#     abs_path = os.getcwd()
#     dir_name = os.path.join(abs_path)
#     f = os.path.join(dir_name, "ms_init_file.csv")
#     data = pd.read_csv(f, usecols=['0', '1', '2', '3', '4', '5'])
#     s_init = np.array(data).tolist()
#     return s_init

def run_payoff_matrix_task():
    qmatrix = [[0.9, 0.05, 0.05], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45],
               [0.9, 0.05, 0.05], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45],
               [0.9, 0.05, 0.05], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45]]
    f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2])
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2])
    f_q = f_q.reshape(f_q.size, 1).transpose()
    expected_payoff_matrix = []
    for p0 in [1, 0]:
        for p1 in [1, 0]:
            for p2 in [1, 0]:
                expected_payoff_matrix.append([])
                for q0 in [1, 0]:
                    for q1 in [1, 0]:
                        for q2 in [1, 0]:
                            pl = [p0, p1, p2]
                            ql = [q0, q1, q2]
                            v, r_p, r_q = calc_expected_payoff(qmatrix, pl, ql, f_p, f_q)
                            expected_payoff_matrix[-1].append((np.round(r_p, 3), np.round(r_q, 3)))
    print(expected_payoff_matrix)
    expected_payoff_matrix_pd = pd.DataFrame(expected_payoff_matrix)
    abs_path = os.path.abspath(os.getcwd())
    print(abs_path)
    csv_file_name = "/expected_payoff_matrix.csv"
    file_name = abs_path + csv_file_name
    expected_payoff_matrix_pd.to_csv(file_name, index=None)


if __name__ == '__main__':
    # s_init_list = read_s_init()
    # init_num = len(s_init_list)
    # p_rd = Pool()
    # for _ in range(init_num):
    #     s_init = s_init_list[_][:]
    #     p_rd.apply_async(run_task_rd, args=(s_init,))
    # p_rd.close()
    # p_rd.join()
    # print("All subprocesses done")
    # # run_task_rd(s_init_list[0])
    # # run_task()
    run_payoff_matrix_task()