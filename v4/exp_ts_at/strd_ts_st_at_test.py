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
    # m = np.array([[qvec[0] * p[0] * q[0], qvec[0] * p[0] * (1 - q[0]), qvec[0] * (1 - p[0]) * q[0],
    #                qvec[0] * (1 - p[0]) * (1 - q[0]),
    #                (1 - qvec[0]) * p[4] * q[4], (1 - qvec[0]) * p[4] * (1 - q[4]), (1 - qvec[0]) * (1 - p[4]) * q[4],
    #                (1 - qvec[0]) * (1 - p[4]) * (1 - q[4])],
    #               [qvec[1] * p[1] * q[1], qvec[1] * p[1] * (1 - q[1]), qvec[1] * (1 - p[1]) * q[1],
    #                qvec[1] * (1 - p[1]) * (1 - q[1]),
    #                (1 - qvec[1]) * p[5] * q[5], (1 - qvec[1]) * p[5] * (1 - q[5]), (1 - qvec[1]) * (1 - p[5]) * q[5],
    #                (1 - qvec[1]) * (1 - p[5]) * (1 - q[5])],
    #               [qvec[2] * p[2] * q[2], qvec[2] * p[2] * (1 - q[2]), qvec[2] * (1 - p[2]) * q[2],
    #                qvec[2] * (1 - p[2]) * (1 - q[2]),
    #                (1 - qvec[2]) * p[6] * q[6], (1 - qvec[2]) * p[6] * (1 - q[6]), (1 - qvec[2]) * (1 - p[6]) * q[6],
    #                (1 - qvec[2]) * (1 - p[6]) * (1 - q[6])],
    #               [qvec[3] * p[3] * q[3], qvec[3] * p[3] * (1 - q[3]), qvec[3] * (1 - p[3]) * q[3],
    #                qvec[3] * (1 - p[3]) * (1 - q[3]),
    #                (1 - qvec[3]) * p[7] * q[7], (1 - qvec[3]) * p[7] * (1 - q[7]), (1 - qvec[3]) * (1 - p[7]) * q[7],
    #                (1 - qvec[3]) * (1 - p[7]) * (1 - q[7])],
    #               [qvec[4] * p[0] * q[0], qvec[4] * p[0] * (1 - q[0]), qvec[4] * (1 - p[0]) * q[0],
    #                qvec[4] * (1 - p[0]) * (1 - q[0]),
    #                (1 - qvec[4]) * p[4] * q[4], (1 - qvec[4]) * p[4] * (1 - q[4]), (1 - qvec[4]) * (1 - p[4]) * q[4],
    #                (1 - qvec[4]) * (1 - p[4]) * (1 - q[4])],
    #               [qvec[5] * p[1] * q[1], qvec[5] * p[1] * (1 - q[1]), qvec[5] * (1 - p[1]) * q[1],
    #                qvec[5] * (1 - p[1]) * (1 - q[1]),
    #                (1 - qvec[5]) * p[5] * q[5], (1 - qvec[5]) * p[5] * (1 - q[5]), (1 - qvec[5]) * (1 - p[5]) * q[5],
    #                (1 - qvec[5]) * (1 - p[5]) * (1 - q[5])],
    #               [qvec[6] * p[2] * q[2], qvec[6] * p[2] * (1 - q[2]), qvec[6] * (1 - p[2]) * q[2],
    #                qvec[6] * (1 - p[2]) * (1 - q[2]),
    #                (1 - qvec[6]) * p[6] * q[6], (1 - qvec[6]) * p[6] * (1 - q[6]), (1 - qvec[6]) * (1 - p[6]) * q[6],
    #                (1 - qvec[6]) * (1 - p[6]) * (1 - q[6])],
    #               [qvec[7] * p[3] * q[3], qvec[7] * p[3] * (1 - q[3]), qvec[7] * (1 - p[3]) * q[3],
    #                qvec[7] * (1 - p[3]) * (1 - q[3]),
    #                (1 - qvec[7]) * p[7] * q[7], (1 - qvec[7]) * p[7] * (1 - q[7]), (1 - qvec[7]) * (1 - p[7]) * q[7],
    #                (1 - qvec[7]) * (1 - p[7]) * (1 - q[7])]])
    m = np.ones((8, 8))
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
    return m


def calc_expected_payoff(qvec, pl, ql, f_p, f_q):
    pa = pl[:]
    qa = ql[:]
    m = build_markov_chain(qvec, pa, qa)
    # print(m)
    null_matrix = np.transpose(m) - np.eye(8)
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
        null_matrix = np.transpose(m) - np.eye(8)
        v = null_space(null_matrix)
        v = v / np.sum(v)
        action_payoff = np.dot(f_p, v)[0][0]
        return v, action_payoff
    elif agent_id == 1:
        pa = pl[:]
        qa = ql[:]
        pa[s] = a_p
        qa[s] = a_q
        m = build_markov_chain(qvec, pa, qa)
        null_matrix = np.transpose(m) - np.eye(8)
        v = null_space(null_matrix)
        v = v / np.sum(v)
        action_payoff = np.dot(f_q, v)[0][0]
        return v, action_payoff


# def calc_payoff_sp(agent_id, s, qvec, pl, ql, f_p, f_q):
#     if agent_id == 0:
#         m_cc = build_markov_chain(qvec, [1, pl[1]], [1, ql[1]])
#         m_cd = build_markov_chain(qvec, [1, pl[1]], [0, ql[1]])
#         m_dc = build_markov_chain(qvec, [0, pl[1]], [1, ql[1]])
#         m_dd = build_markov_chain(qvec, [0, pl[1]], [0, ql[1]])
#         m_c = m_cc * ql[0] + m_cd * (1 - ql[0])
#         m_d = m_dc * ql[0] + m_dd * (1 - ql[0])
#         m = pl[0] * m_c + (1 - pl[0]) * m_d
#         null_matrix = np.transpose(m) - np.eye(8)
#         v = null_space(null_matrix)
#         v = v / np.sum(v)
#         action_payoff = np.dot(f_p, v)[0][0]
#         return action_payoff
#
#
# def calc_payoff_sp_c(agent_id, s, qvec, pl, ql, f_p, f_q):
#     if agent_id == 0:
#         m_cc = build_markov_chain(qvec, [1, pl[1]], [1, ql[1]])
#         m_cd = build_markov_chain(qvec, [1, pl[1]], [0, ql[1]])
#         m_dc = build_markov_chain(qvec, [0, pl[1]], [1, ql[1]])
#         m_dd = build_markov_chain(qvec, [0, pl[1]], [0, ql[1]])
#         m_c = m_cc * ql[0] + m_cd * (1 - ql[0])
#         null_matrix = np.transpose(m_c) - np.eye(8)
#         v = null_space(null_matrix)
#         v = v / np.sum(v)
#         action_payoff = np.dot(f_p, v)[0][0]
#         return action_payoff



# def run_task():
#     p0 = 0.4
#     p1 = 0.5
#     q0 = 0.7
#     q1 = 0.6
#     z_1 = 0.9
#     z_2 = 0.1
#     qvec = [z_1, z_2, z_2, z_2, z_1, z_2, z_2, z_2]
#     f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2])
#     # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
#     f_p = f_p.reshape(f_p.size, 1).transpose()
#     f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2])
#     # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
#     f_q = f_q.reshape(f_q.size, 1).transpose()
#     pl = [p0, p0, p0, p0, p1, p1, p1, p1]
#     ql = [q0, q0, q0, q0, q1, q1, q1, q1]
#     payoff = calc_payoff(0, 0, 1, qvec=qvec, pl=pl, ql=ql, f_p=f_p, f_q=f_q)
#     print(payoff)


# def average_game(s_n, qvec, pl, ql, f_p, f_q):
#     average_payoff = []
#     for i in range(s_n):
#         average_payoff.append([[0 for _ in range(4)], [0 for _ in range(4)]])
#     for s_i in range(s_n):
#         for a_i in [1, 0]:
#             for a_j in [1, 0]:
#                 pa = pl[:]
#                 qa = ql[:]
#                 for s_j in range(4):
#                     pa[s_i * 4 + s_j] = a_i
#                     qa[s_i * 4 + s_j] = a_j
#                 m = build_markov_chain(qvec, pa, qa)
#                 null_matrix = np.transpose(m) - np.eye(8)
#                 v = null_space(null_matrix)
#                 v = v / np.sum(v)
#                 r_p = np.dot(f_p, v)[0][0]
#                 r_q = np.dot(f_q, v)[0][0]
#                 average_payoff[s_i][0][(1 - a_i) * 2 + (1 - a_j)] = r_p
#                 average_payoff[s_i][1][(1 - a_i) * 2 + (1 - a_j)] = r_q
#     m = build_markov_chain(qvec, pl, ql)
#     null_matrix = np.transpose(m) - np.eye(8)
#     v = null_space(null_matrix)
#     v = v / np.sum(v)
#     v = np.array(v)
#     v = v.transpose()[0]
#     return v, average_payoff
#
#
# def build_payoff_matrix(s_i, average_payoff, id='p'):
#     if id == 'p':
#         apm = np.array([[average_payoff[s_i][0][0], average_payoff[s_i][0][1]],
#                         [average_payoff[s_i][0][2], average_payoff[s_i][0][3]]])
#     else:
#         apm = np.array([[average_payoff[s_i][1][0], average_payoff[s_i][1][2]],
#                         [average_payoff[s_i][1][1], average_payoff[s_i][1][3]]])
#     return apm
#

# def evolve(s_n, average_payoff, p0, q0, p1, q1, step_size, v):
#     for s_i in range(s_n):
#         p_m = build_payoff_matrix(s_i, average_payoff, id='p')
#         q_m = build_payoff_matrix(s_i, average_payoff, id='q')
#         if s_i == 0:
#             p_o = np.dot(p_m, [[q0], [1 - q0]])
#             q_o = np.dot(q_m, [[p0], [1 - p0]])
#             v_s = np.sum(v[0:4])
#             dp = (np.dot([1, 0], p_o)[0] - np.dot([p0, 1 - p0], p_o)[0]) * p0 * v_s
#             dq = (np.dot([1, 0], q_o)[0] - np.dot([q0, 1 - q0], q_o)[0]) * q0 * v_s
#             p0 = p0 + dp * step_size
#             q0 = q0 + dq * step_size
#         else:
#             p_o = np.dot(p_m, [[q1], [1 - q1]])
#             q_o = np.dot(q_m, [[p1], [1 - p1]])
#             v_s = np.sum(v[4:8])
#             dp = (np.dot([1, 0], p_o)[0] - np.dot([p1, 1 - p1], p_o)[0]) * p1 * v_s
#             dq = (np.dot([1, 0], q_o)[0] - np.dot([q1, 1 - q1], q_o)[0]) * q1 * v_s
#             p1 = p1 + dp * step_size
#             q1 = q1 + dq * step_size
#     return p0, q0, p1, q1


def run_task_rd(s_init):
    # t = np.arange(0, int(10e5))
    t = range(0, 1)
    step_size = 0.001
    s_n = 2
    print(s_init)
    for z_1, z_2 in [[0.9, 0.1]]:
        p0 = s_init[0]
        q0 = s_init[1]
        p1 = s_init[2]
        q1 = s_init[3]
        print(z_1, z_2)
        # qvec = [z_1, z_2, z_2, z_2, z_1, z_2, z_2, z_2]
        qmatrix = [[0.8, 0.2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9],
                   [0.8, 0.2], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]
        qmatrix0 = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                    [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        qmatrix1 = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
                    [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2])
        # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
        f_p = f_p.reshape(f_p.size, 1).transpose()
        f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2])
        # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
        f_q = f_q.reshape(f_q.size, 1).transpose()
        d = []
        d.append([p0, q0, p1, q1])
        for _ in t:
            if _ % 1000 == 0:
                print('strd', _)
            pl = [p0, p1]
            ql = [q0, q1]
            # pl = [p0, p0, p0, p0, p1, p1, p1, p1]
            # ql = [q0, q0, q0, q0, q1, q1, q1, q1]
            # calc_expected_payoff(qvec, pl, ql, f_p, f_q)
            v, r_p_e, r_q_e = calc_expected_payoff(qmatrix, pl, ql, f_p, f_q)
            # calc_payoff(agent_id, s, a_p, a_q, qvec, pl, ql, f_p, f_q)
            v_p_0_cc, r_p_0_cc = calc_payoff(0, 0, 1, 1, qmatrix, pl, ql, f_p, f_q)
            v_q_0_cc, r_q_0_cc = calc_payoff(1, 0, 1, 1, qmatrix, pl, ql, f_p, f_q)
            v_p_0_cd, r_p_0_cd = calc_payoff(0, 0, 1, 0, qmatrix, pl, ql, f_p, f_q)
            v_q_0_cd, r_q_0_cd = calc_payoff(1, 0, 1, 0, qmatrix, pl, ql, f_p, f_q)
            v_p_0_dc, r_p_0_dc = calc_payoff(0, 0, 0, 1, qmatrix, pl, ql, f_p, f_q)
            v_q_0_dc, r_q_0_dc = calc_payoff(1, 0, 0, 1, qmatrix, pl, ql, f_p, f_q)
            v_p_0_dd, r_p_0_dd = calc_payoff(0, 0, 0, 0, qmatrix, pl, ql, f_p, f_q)
            v_q_0_dd, r_q_0_dd = calc_payoff(1, 0, 0, 0, qmatrix, pl, ql, f_p, f_q)
            v_p_1_cc, r_p_1_cc = calc_payoff(0, 1, 1, 1, qmatrix, pl, ql, f_p, f_q)
            v_q_1_cc, r_q_1_cc = calc_payoff(1, 1, 1, 1, qmatrix, pl, ql, f_p, f_q)
            v_p_1_cd, r_p_1_cd = calc_payoff(0, 1, 1, 0, qmatrix, pl, ql, f_p, f_q)
            v_q_1_cd, r_q_1_cd = calc_payoff(1, 1, 1, 0, qmatrix, pl, ql, f_p, f_q)
            v_p_1_dc, r_p_1_dc = calc_payoff(0, 1, 0, 1, qmatrix, pl, ql, f_p, f_q)
            v_q_1_dc, r_q_1_dc = calc_payoff(1, 1, 0, 1, qmatrix, pl, ql, f_p, f_q)
            v_p_1_dd, r_p_1_dd = calc_payoff(0, 1, 0, 0, qmatrix, pl, ql, f_p, f_q)
            v_q_1_dd, r_q_1_dd = calc_payoff(1, 1, 0, 0, qmatrix, pl, ql, f_p, f_q)
            v_0 = np.sum(v[0:4])
            v_1 = np.sum(v[4:8])
            f_p_exp = np.array([r_p_0_cc, r_p_0_cd, r_p_0_dc, r_p_0_dd, r_p_1_cc, r_p_1_cd, r_p_1_dc, r_p_1_dd])
            f_p_exp = f_p_exp.reshape(f_p_exp.size, 1).transpose()
            f_q_exp = np.array([r_q_0_cc, r_q_0_cd, r_q_0_dc, r_q_0_dd, r_q_1_cc, r_q_1_cd, r_q_1_dc, r_q_1_dd])
            f_q_exp = f_q_exp.reshape(f_q_exp.size, 1).transpose()
            # dp0 = ((r_p_0_cc * q0 + r_p_0_cd * (1 - q0)) - (p0 * (r_p_0_cc * q0 + r_p_0_cd * (1 - q0)) + (1 - p0) * (
            #         r_p_0_dc * q0 + r_p_0_dd * (1 - q0)))) * p0 * v_0
            # dq0 = ((r_q_0_cc * p0 + r_q_0_dc * (1 - p0)) - (q0 * (r_q_0_cc * p0 + r_q_0_dc * (1 - p0)) + (1 - q0) * (
            #         r_q_0_cd * p0 + r_q_0_dd * (1 - p0)))) * q0 * v_0
            # dp1 = ((r_p_1_cc * q1 + r_p_1_cd * (1 - q1)) - (p1 * (r_p_1_cc * q1 + r_p_1_cd * (1 - q1)) + (1 - p1) * (
            #         r_p_1_dc * q1 + r_p_1_dd * (1 - q1)))) * p1 * v_1
            # dq1 = ((r_q_1_cc * p1 + r_q_1_dc * (1 - p1)) - (q1 * (r_q_1_cc * p1 + r_q_1_dc * (1 - p1)) + (1 - q1) * (
            #         r_q_1_cd * p1 + r_q_1_dd * (1 - p1)))) * q1 * v_1
            # p_0_c = r_p_0_cc * q0 + r_p_0_cd * (1 - q0)
            # p_0_c_sp = calc_payoff_sp_c(0, 0, qmatrix, pl, ql, f_p, f_q)
            # p_0 = p0 * (r_p_0_cc * q0 + r_p_0_cd * (1 - q0)) + (1 - p0) * (r_p_0_dc * q0 + r_p_0_dd * (1 - q0))
            # p_0_sp = calc_payoff_sp(0, 0, qmatrix, pl, ql, f_p, f_q)
            # print(p_0_c - p_0, p_0_c_sp - p_0_sp)
            print(v_p_0_cc)
            print(v_p_0_cd)
            dp0 = (calc_payoff(0, 0, 1, q0, qmatrix0, pl, ql, f_p_exp, f_q_exp)[1] - calc_payoff(0, 0, p0, q0, qmatrix0,
                                                                                              pl, ql,
                                                                                              f_p_exp,
                                                                                              f_q_exp)[1]) * p0 * v_0
            print(dp0)
            dq0 = (calc_payoff(1, 0, p0, 1, qmatrix0, pl, ql, f_p_exp, f_q_exp)[1] - calc_payoff(1, 0, p0, q0, qmatrix0,
                                                                                              pl, ql,
                                                                                              f_p_exp,
                                                                                              f_q_exp)[1]) * q0 * v_0
            dp1 = (calc_payoff(0, 1, 1, q1, qmatrix1, pl, ql, f_p_exp, f_q_exp)[1] - calc_payoff(0, 1, p1, q1, qmatrix1,
                                                                                              pl, ql,
                                                                                              f_p_exp,
                                                                                              f_q_exp)[1]) * p1 * v_1
            dq1 = (calc_payoff(1, 1, p1, 1, qmatrix1, pl, ql, f_p_exp, f_q_exp)[1] - calc_payoff(1, 1, p1, q1, qmatrix1,
                                                                                              pl, ql,
                                                                                              f_p_exp,
                                                                                              f_q_exp)[1]) * q1 * v_1
            p0 = p0 + dp0 * step_size
            q0 = q0 + dq0 * step_size
            p1 = p1 + dp1 * step_size
            q1 = q1 + dq1 * step_size
            d.append([p0, q0, p1, q1])
        # abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results_st_at"))
        # csv_file_name = "/strd_ts_st_at_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (
        # s_init[0], s_init[1], s_init[2], s_init[3])
        # file_name = abs_path + csv_file_name
        # d_pd = pd.DataFrame(d)
        # d_pd.to_csv(file_name, index=None)


def read_s_init():
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "s_init_file.csv")
    data = pd.read_csv(f, usecols=['0', '1', '2', '3'])
    s_init = np.array(data).tolist()
    return s_init


if __name__ == '__main__':
    s_init_list = read_s_init()
    # init_num = len(s_init_list)
    # p_rd = Pool()
    # for _ in range(init_num):
    #     s_init = s_init_list[_][:]
    #     p_rd.apply_async(run_task_rd, args=(s_init,))
    # p_rd.close()
    # p_rd.join()
    # print("All subprocesses done")
    run_task_rd(s_init_list[6])
    # run_task()
