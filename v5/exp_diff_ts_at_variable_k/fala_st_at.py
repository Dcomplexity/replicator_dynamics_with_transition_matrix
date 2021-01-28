import numpy as np
import random
import pandas as pd
import os
import math
from copy import deepcopy
from sympy import Matrix
from scipy.linalg import null_space
from multiprocessing import Pool

def build_markov_chain(qvec, p, q):
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
    null_matrix = np.transpose(m) - np.eye(8)
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
        null_matrix = np.transpose(m) - np.eye(8)
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
        null_matrix = np.transpose(m) - np.eye(8)
        v = null_space(null_matrix)
        v = v / np.sum(v)
        action_payoff = np.dot(f_q, v)[0][0]
        v_0 = np.sum(v[0:4])
        return action_payoff, v_0


def clac_k(p0, q0, p1, q1):
    qmatrix = [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9],
               [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]

    f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2])
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2])
    f_q = f_q.reshape(f_q.size, 1).transpose()

    pl = [p0, p1]
    ql = [q0, q1]
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
    v_0 = np.sum(v[0:4])
    v_1 = np.sum(v[4:8])
    k = 0
    k_p_0 = 0
    k_q_0 = 0
    k_p_1 = 0
    k_q_1 = 0
    dp0_c_d = ((r_p_0_cc * q0 + r_p_0_cd * (1 - q0)) - (r_p_0_dc * q0 + r_p_0_dd * (1 - q0)))
    vp0_c_d = ((v_p_0_cc * q0 + v_p_0_cd * (1 - q0)) - (v_p_0_dc * q0 + v_p_0_dd * (1 - q0)))
    dq0_c_d = ((r_q_0_cc * p0 + r_q_0_dc * (1 - p0)) - (r_q_0_cd * p0 + r_q_0_dd * (1 - p0)))
    vq0_c_d = ((v_q_0_cc * p0 + v_q_0_dc * (1 - p0)) - (v_q_0_cd * p0 + v_q_0_dd * (1 - p0)))
    dp1_c_d = ((r_p_1_cc * q1 + r_p_1_cd * (1 - q1)) - (r_p_1_dc * q1 + r_p_1_dd * (1 - q1)))
    vp1_c_d = ((v_p_1_cc * q1 + v_p_1_cd * (1 - q1)) - (v_p_1_dc * q1 + v_p_1_dd * (1 - q1)))
    dq1_c_d = ((r_q_1_cc * p1 + r_q_1_dc * (1 - p1)) - (r_q_1_cd * p1 + r_q_1_dd * (1 - p1)))
    vq1_c_d = ((v_q_1_cc * p1 + v_q_1_dc * (1 - p1)) - (v_q_1_cd * p1 + v_q_1_dd * (1 - p1)))
    if dp0_c_d <= 0 < vp0_c_d:
        k_p_0 = (-1 * dp0_c_d) / vp0_c_d + 0.01
    if dq0_c_d <= 0 < vq0_c_d:
        k_q_0 = (-1 * dq0_c_d) / vq0_c_d + 0.01
    if dp1_c_d <= 0 < vp1_c_d:
        k_p_1 = (-1 * dp1_c_d) / vp1_c_d + 0.01
    if dq1_c_d <= 0 < vq1_c_d:
        k_q_1 = (-1 * dq1_c_d) / vq1_c_d + 0.01
    k = max(k_p_0, k_q_0, k_p_1, k_q_1, 0)
    return k


# pd_game_1 = [[2, 2], [4, 1], [1, 4], [3, 3]]
# pd_game_2 = [[2, 2], [4, 1], [1, 4], [3, 3]]

transition_prob = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1],
                   [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]]

def play_pd_game_1(a_x, a_y, k):
    pd_game_1_k = [[2 + k, 2 + k], [4 + k, 1 + k], [1 + k, 4 + k], [3 + k, 3 + k]]
    return pd_game_1_k[a_x * 2 + a_y]


def play_pd_game_2(a_x, a_y):
    pd_game_2 = [[2, 2], [4, 1], [1, 4], [3, 3]]
    return pd_game_2[a_x * 2 + a_y]


def next_state(s, a_x, a_y):
    prob = transition_prob[s * 4 + a_x * 2 + a_y]
    s_ = np.random.choice([0, 1], p=prob)
    return s_

# def valid_s(s_value):
#     if s_value < 0.0:
#         s_new = 0.0
#     elif s_value > 1.0:
#         s_new = 1.0
#     else:
#         s_new = s_value
#     return s_new


class Agent:
    def __init__(self, alpha, agent_id):
        self.id = agent_id
        self.strategy = []
        self.actions = [0, 1]  # 0 for defection and 1 for cooperation
        self.states = [0, 1]  # there are two state: 0 and 1
        self.len_action = len(self.actions)
        self.alpha = alpha
        self.strategy_trace = []

    def get_strategy(self):
        return self.strategy

    def initial_strategy(self):
        for state in self.states:
            self.strategy.append([1 / self.len_action for _ in range(self.len_action)])

    def set_strategy(self, new_stra=None):
        """
        set a specified strategy
        :param new_s: the probability to play cooperation (action[1])
        :return:
        """
        if new_stra:
            for state in self.states:
                self.strategy[state] = new_stra[state]

    def choose_action(self, s):
        # print(self.id, self.strategy)
        a = np.random.choice(np.array(self.actions), size=1, p=self.strategy[s])[0]
        return a

    def update_strategy(self, s, a, r):
        for action in self.actions:
            if action == a:
                self.strategy[s][action] = self.strategy[s][action] + self.alpha * r * (1 - self.strategy[s][action])
            else:
                self.strategy[s][action] = self.strategy[s][action] - self.alpha * r * self.strategy[s][action]

    def record_strategy(self):
        self.strategy_trace.append(deepcopy(self.strategy))


def run_game_fala(agent_x_init_strategy, agent_y_init_strategy, s_0):
    agent_x = Agent(alpha=0.00001, agent_id=0)
    agent_y = Agent(alpha=0.00001, agent_id=1)
    agent_x.initial_strategy()
    agent_y.initial_strategy()
    agent_x.set_strategy(agent_x_init_strategy)
    agent_y.set_strategy(agent_y_init_strategy)
    cur_s = s_0
    games = [play_pd_game_1, play_pd_game_2]
    r_sum_x = np.array([0.0, 0.0])  # store the sum of reward of agent_x for each state: state 0 and state 1
    r_sum_y = np.array([0.0, 0.0])
    tau_x = np.array([0.0, 0.0])  # store the average reward of agent_x for each state: state 0 and state 1
    tau_y = np.array([0.0, 0.0])
    action_t_x = np.array([0, 0])  # for agent_x, the last action taken in state 0 and state 1
    action_t_y = np.array([0, 0])
    time_step = np.array([0, 0])  # store the sum of time since last time meeting each state: state 0 and state 1
    visited = [0, 0]
    p = []
    for _ in range(int(2 * 10e6)):
        if _ % 10000 == 0:
            print('fala', _)
        # agent_x.record_strategy()
        # agent_y.record_strategy()
        p.append([agent_x.strategy[0][1], agent_y.strategy[0][1], agent_x.strategy[1][1], agent_y.strategy[1][1]])
        p0 = agent_x.strategy[0][1]
        q0 = agent_y.strategy[0][1]
        p1 = agent_x.strategy[1][1]
        q1 = agent_y.strategy[1][1]
        if visited[cur_s] == 0:
            visited[cur_s] = 1
            r_sum_x[cur_s] = 0
            r_sum_y[cur_s] = 0
            time_step[cur_s] = 0
            a_x = agent_x.choose_action(cur_s)
            a_y = agent_y.choose_action(cur_s)
            action_t_x[cur_s] = a_x
            action_t_y[cur_s] = a_y
            r_x, r_y = games[cur_s](a_x, a_y)
            r_sum_x = r_sum_x + r_x
            r_sum_y = r_sum_y + r_y
            time_step = time_step + 1
            cur_s = next_state(cur_s, a_x, a_y)
        else:
            tau_x[cur_s] = r_sum_x[cur_s] / time_step[cur_s]
            tau_y[cur_s] = r_sum_y[cur_s] / time_step[cur_s]
            agent_x.update_strategy(cur_s, action_t_x[cur_s], tau_x[cur_s])
            agent_y.update_strategy(cur_s, action_t_y[cur_s], tau_y[cur_s])
            r_sum_x[cur_s] = 0
            r_sum_y[cur_s] = 0
            time_step[cur_s] = 0
            a_x = agent_x.choose_action(cur_s)
            a_y = agent_y.choose_action(cur_s)
            action_t_x[cur_s] = a_x
            action_t_y[cur_s] = a_y
            r_x, r_y = games[cur_s](a_x, a_y)
            r_sum_x = r_sum_x + r_x
            r_sum_y = r_sum_y + r_y
            time_step = time_step + 1
            cur_s = next_state(cur_s, a_x, a_y)

    # p = []
    # for i in range(len(agent_x.strategy_trace)):
    #     p.append([agent_x.strategy_trace[i][0][1], agent_y.strategy_trace[i][0][1], agent_x.strategy_trace[i][1][1],
    #               agent_y.strategy_trace[i][1][1]])
    # return p, agent_x.strategy_trace, agent_y.strategy_trace
    return p

def run_task_fala(s_init):
    p0 = s_init[0]
    q0 = s_init[1]
    p1 = s_init[2]
    q1 = s_init[3]
    p = run_game_fala(agent_x_init_strategy=[[1 - p0, p0], [1 - p1, p1]],
                                  agent_y_init_strategy=[[1 - q0, q0], [1 - q1, q1]], s_0=0)
    abs_path = os.path.abspath(os.path.join(os.getcwd(), "results_st_at"))
    csv_file_name = "/fala_ts_st_at_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (p0, q0, p1, q1)
    file_name = abs_path + csv_file_name
    d_pd = pd.DataFrame(p)
    d_pd.to_csv(file_name, index=None)


def read_s_init():
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "s_init_file.csv")
    data = pd.read_csv(f, usecols=['0', '1', '2', '3'])
    s_init = np.array(data).tolist()
    return s_init


if __name__ == '__main__':
    p_fala = Pool()
    s_init_list = read_s_init()
    init_num = len(s_init_list)
    for _ in range(init_num):
        s_init = s_init_list[_][:]
        p_fala.apply_async(run_task_fala, args=(s_init,))
    p_fala.close()
    p_fala.join()
    print("All subprocesses done")
