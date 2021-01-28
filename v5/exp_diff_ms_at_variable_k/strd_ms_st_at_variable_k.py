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
    return m


def calc_expected_payoff(qvec, pl, ql, f_p, f_q):
    pa = pl[:]
    qa = ql[:]
    m = build_markov_chain(qvec, pa, qa)
    null_matrix = np.transpose(m) - np.eye(12)
    v = null_space(null_matrix)
    v = v / np.sum(v)
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
        v_0 = np.sum(v[0:4])
        return action_payoff, v_0
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
        v_0 = np.sum(v[0:4])
        return action_payoff, v_0


def run_task_rd(s_init):
    t = np.arange(0, int(2 * 10e5))
    step_size = 0.01
    s_n = 3
    print(s_init)
    for z_1, z_2 in [[0.9, 0.1]]:
        p0 = s_init[0]
        q0 = s_init[1]
        p1 = s_init[2]
        q1 = s_init[3]
        p2 = s_init[4]
        q2 = s_init[5]
        print(z_1, z_2)
        # qvec = [z_1, z_2, z_2, z_2, z_1, z_2, z_2, z_2]
        qmatrix = [[0.9, 0.05, 0.05], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45],
                   [0.9, 0.05, 0.05], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45],
                   [0.9, 0.05, 0.05], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45]]
        # qmatrix0 = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        #             [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        #             [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        # qmatrix1 = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        #             [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        #             [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        # qmatrix2 = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        #             [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        #             [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2])
        # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
        f_p = f_p.reshape(f_p.size, 1).transpose()
        f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2])
        # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
        f_q = f_q.reshape(f_q.size, 1).transpose()
        d = []
        d.append([p0, q0, p1, q1, p2, q2])
        k_list = []
        for _ in t:
            # print('strd', _)
            if _ % 1000 == 0:
                print('strd', _)
            pl = [p0, p1, p2]
            ql = [q0, q1, q2]
            v, r_p_e, r_q_e = calc_expected_payoff(qmatrix, pl, ql, f_p, f_q)
            r_p_0_cc, v_p_0_cc = calc_payoff(0, 0, 1, 1, qmatrix, pl, ql, f_p, f_q)
            r_q_0_cc, v_q_0_cc = calc_payoff(1, 0, 1, 1, qmatrix, pl, ql, f_p, f_q)
            r_p_0_cd, v_p_0_cd = calc_payoff(0, 0, 1, 0, qmatrix, pl, ql, f_p, f_q)
            r_q_0_cd, v_q_0_cd = calc_payoff(1, 0, 1, 0, qmatrix, pl, ql, f_p, f_q)
            r_p_0_dc, v_p_0_dc = calc_payoff(0, 0, 0, 1, qmatrix, pl, ql, f_p, f_q)
            r_q_0_dc, v_q_0_dc = calc_payoff(1, 0, 0, 1, qmatrix, pl, ql, f_p, f_q)
            r_p_0_dd, v_p_0_dd = calc_payoff(0, 0, 0, 0, qmatrix, pl, ql, f_p, f_q)
            r_q_0_dd, v_q_0_dd = calc_payoff(1, 0, 0, 0, qmatrix, pl, ql, f_p, f_q)
            r_p_1_cc, v_p_1_cc = calc_payoff(0, 1, 1, 1, qmatrix, pl, ql, f_p, f_q)
            r_q_1_cc, v_q_1_cc = calc_payoff(1, 1, 1, 1, qmatrix, pl, ql, f_p, f_q)
            r_p_1_cd, v_p_1_cd = calc_payoff(0, 1, 1, 0, qmatrix, pl, ql, f_p, f_q)
            r_q_1_cd, v_q_1_cd = calc_payoff(1, 1, 1, 0, qmatrix, pl, ql, f_p, f_q)
            r_p_1_dc, v_p_1_dc = calc_payoff(0, 1, 0, 1, qmatrix, pl, ql, f_p, f_q)
            r_q_1_dc, v_q_1_dc = calc_payoff(1, 1, 0, 1, qmatrix, pl, ql, f_p, f_q)
            r_p_1_dd, v_p_1_dd = calc_payoff(0, 1, 0, 0, qmatrix, pl, ql, f_p, f_q)
            r_q_1_dd, v_q_1_dd = calc_payoff(1, 1, 0, 0, qmatrix, pl, ql, f_p, f_q)
            r_p_2_cc, v_p_2_cc = calc_payoff(0, 2, 1, 1, qmatrix, pl, ql, f_p, f_q)
            r_q_2_cc, v_q_2_cc = calc_payoff(1, 2, 1, 1, qmatrix, pl, ql, f_p, f_q)
            r_p_2_cd, v_p_2_cd = calc_payoff(0, 2, 1, 0, qmatrix, pl, ql, f_p, f_q)
            r_q_2_cd, v_q_2_cd = calc_payoff(1, 2, 1, 0, qmatrix, pl, ql, f_p, f_q)
            r_p_2_dc, v_p_2_dc = calc_payoff(0, 2, 0, 1, qmatrix, pl, ql, f_p, f_q)
            r_q_2_dc, v_q_2_dc = calc_payoff(1, 2, 0, 1, qmatrix, pl, ql, f_p, f_q)
            r_p_2_dd, v_p_2_dd = calc_payoff(0, 2, 0, 0, qmatrix, pl, ql, f_p, f_q)
            r_q_2_dd, v_q_2_dd = calc_payoff(1, 2, 0, 0, qmatrix, pl, ql, f_p, f_q)
            v_0 = np.sum(v[0:4])
            v_1 = np.sum(v[4:8])
            v_2 = np.sum(v[8:12])
            k_p_0 = 0
            k_q_0 = 0
            k_p_1 = 0
            k_q_1 = 0
            k_p_2 = 0
            k_q_2 = 0
            dp0_c_d = ((r_p_0_cc * q0 + r_p_0_cd * (1 - q0)) - (r_p_0_dc * q0 + r_p_0_dd * (1 - q0)))
            vp0_c_d = ((v_p_0_cc * q0 + v_p_0_cd * (1 - q0)) - (v_p_0_dc * q0 + v_p_0_dd * (1 - q0)))
            dq0_c_d = ((r_q_0_cc * p0 + r_q_0_dc * (1 - p0)) - (r_q_0_cd * p0 + r_q_0_dd * (1 - p0)))
            vq0_c_d = ((v_q_0_cc * p0 + v_q_0_dc * (1 - p0)) - (v_q_0_cd * p0 + v_q_0_dd * (1 - p0)))
            dp1_c_d = ((r_p_1_cc * q1 + r_p_1_cd * (1 - q1)) - (r_p_1_dc * q1 + r_p_1_dd * (1 - q1)))
            vp1_c_d = ((v_p_1_cc * q1 + v_p_1_cd * (1 - q1)) - (v_p_1_dc * q1 + v_p_1_dd * (1 - q1)))
            dq1_c_d = ((r_q_1_cc * p1 + r_q_1_dc * (1 - p1)) - (r_q_1_cd * p1 + r_q_1_dd * (1 - p1)))
            vq1_c_d = ((v_q_1_cc * p1 + v_q_1_dc * (1 - p1)) - (v_q_1_cd * p1 + v_q_1_dd * (1 - p1)))
            dp2_c_d = ((r_p_2_cc * q2 + r_p_2_cd * (1 - q2)) - (r_p_2_dc * q2 + r_p_2_dd * (1 - q2)))
            vp2_c_d = ((v_p_2_cc * q2 + v_p_2_cd * (1 - q2)) - (v_p_2_dc * q2 + v_p_2_dd * (1 - q2)))
            dq2_c_d = ((r_q_2_cc * p2 + r_q_2_dc * (1 - p2)) - (v_q_2_cd * p2 + v_q_2_dd * (1 - p2)))
            vq2_c_d = ((v_q_2_cc * p2 + v_q_2_dc * (1 - p2)) - (v_q_2_cd * p2 + v_q_2_dd * (1 - p2)))
            if dp0_c_d <= 0 < vp0_c_d:
                k_p_0 = (-1 * dp0_c_d) / vp0_c_d + 0.01
            if dq0_c_d <= 0 < vq0_c_d:
                k_q_0 = (-1 * dq0_c_d) / vq0_c_d + 0.01
            if dp1_c_d <= 0 < vp1_c_d:
                k_p_1 = (-1 * dp1_c_d) / vp1_c_d + 0.01
            if dq1_c_d <= 0 < vq1_c_d:
                k_q_1 = (-1 * dq1_c_d) / vq1_c_d + 0.01
            if dp2_c_d <= 0 < vp2_c_d:
                k_p_2 = (-1 * dp2_c_d) / vp2_c_d + 0.01
            if dq2_c_d <= 0 < vq2_c_d:
                k_q_2 = (-1 * dq2_c_d) / vq2_c_d + 0.01
            k = max(k_p_0, k_q_0, k_p_2, k_q_2, k_p_1, k_q_1, 0)
            dp0 = (dp0_c_d + k * vp0_c_d) * p0 * (1 - p0) * v_0
            dq0 = (dq0_c_d + k * vq0_c_d) * q0 * (1 - q0) * v_0
            dp1 = (dp1_c_d + k * vp1_c_d) * p1 * (1 - p1) * v_1
            dq1 = (dq1_c_d + k * vq1_c_d) * q1 * (1 - q1) * v_1
            dp2 = (dp2_c_d + k * vp2_c_d) * p2 * (1 - p2) * v_2
            dq2 = (dq2_c_d + k * vq2_c_d) * q2 * (1 - q2) * v_2
            p0 = p0 + dp0 * step_size
            q0 = q0 + dq0 * step_size
            p1 = p1 + dp1 * step_size
            q1 = q1 + dq1 * step_size
            p2 = p2 + dp2 * step_size
            q2 = q2 + dq2 * step_size
            d.append([p0, q0, p1, q1, p2, q2])
            k_list.append(k)
        abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results_st_at"))
        csv_file_name_st = "/variable_k_strd_ms_st_at_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (
            s_init[0], s_init[1], s_init[2], s_init[3], s_init[4], s_init[5])
        csv_file_name_k = "/k_value_at_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (
            s_init[0], s_init[1], s_init[2], s_init[3], s_init[4], s_init[5])
        file_name_st = abs_path + csv_file_name_st
        file_name_k = abs_path + csv_file_name_k
        d_pd = pd.DataFrame(d)
        k_pd = pd.DataFrame(k_list)
        d_pd.to_csv(file_name_st, index=None)
        k_pd.to_csv(file_name_k, index=None)


def read_s_init():
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "ms_init_file.csv")
    data = pd.read_csv(f, usecols=['0', '1', '2', '3', '4', '5'])
    s_init = np.array(data).tolist()
    return s_init


if __name__ == '__main__':
    s_init_list = read_s_init()
    init_num = len(s_init_list)
    p_rd = Pool()
    for _ in range(init_num):
        s_init = s_init_list[_][:]
        p_rd.apply_async(run_task_rd, args=(s_init,))
    p_rd.close()
    p_rd.join()
    print("All subprocesses done")
    # run_task_rd(s_init_list[0])
    # run_task()
