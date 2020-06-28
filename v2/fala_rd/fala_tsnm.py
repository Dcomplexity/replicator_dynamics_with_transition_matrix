import numpy as np
import pandas as pd
import os
import math
from copy import deepcopy
from game_env import *
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


def run_game_fala(agent_0_init_strategy, agent_1_init_strategy, s_0):
    agent_0 = Agent(alpha=0.0001, agent_id=0)
    agent_1 = Agent(alpha=0.0001, agent_id=1)
    agent_0.initial_strategy()
    agent_1.initial_strategy()
    agent_0.set_strategy(agent_0_init_strategy)
    agent_1.set_strategy(agent_1_init_strategy)
    cur_s = s_0
    games = [play_pd_game_1, play_pd_game_2]
    r_sum_0 = np.array([0.0, 0.0])  # store the sum of reward of agent_0 for each state: state 0 and state 1
    r_sum_1 = np.array([0.0, 0.0])
    tau_0 = np.array([0.0, 0.0])
    tau_1 = np.array([0.0, 0.0])
    action_t_0 = np.array([0, 0])
    action_t_1 = np.array([0, 0])
    time_step = np.array([0, 0])  # store the sum of time belonging to  each state: state 0 and state 1
    visited = [0, 0]
    for _ in range(10000000):
        agent_0.record_strategy()
        agent_1.record_strategy()
        if visited[cur_s] == 0 or visited[1 - cur_s] == 0:
            visited[cur_s] = 1
            a_0 = agent_0.choose_action(cur_s)
            a_1 = agent_1.choose_action(cur_s)
            action_t_0[cur_s] = a_0
            action_t_1[cur_s] = a_1
            r_0, r_1 = games[cur_s](a_0, a_1)
            r_sum_0[cur_s] = r_sum_0[cur_s] + r_0
            r_sum_1[cur_s] = r_sum_1[cur_s] + r_1
            time_step[cur_s] += 1
            cur_s = next_state(cur_s, a_0, a_1)
        else:
            tau_0[cur_s] = r_sum_0[cur_s] / time_step[cur_s]
            tau_1[cur_s] = r_sum_1[cur_s] / time_step[cur_s]
            agent_0.update_strategy(cur_s, action_t_0[cur_s], tau_0[cur_s])
            agent_1.update_strategy(cur_s, action_t_1[cur_s], tau_1[cur_s])
            r_sum_0[cur_s] = 0
            r_sum_1[cur_s] = 0
            a_0 = agent_0.choose_action(cur_s)
            a_1 = agent_1.choose_action(cur_s)
            action_t_0[cur_s] = a_0
            action_t_1[cur_s] = a_1
            r_0, r_1 = games[cur_s](a_0, a_1)
            r_sum_0 += r_0
            r_sum_1 += r_1
            time_step[cur_s] = 0
            time_step += 1
            cur_s = next_state(cur_s, a_0, a_1)
        # print(len(agent_0.strategy_trace), len(agent_1.strategy_trace))

    p = []
    for i in range(len(agent_0.strategy_trace)):
        p.append([agent_0.strategy_trace[i][0][1], agent_1.strategy_trace[i][0][1], agent_0.strategy_trace[i][1][1],
                  agent_1.strategy_trace[i][1][1]])
    return p, agent_0.strategy_trace, agent_1.strategy_trace

def run_task_fala(p_init):
    p0 = p_init[0]
    q0 = p_init[1]
    p1 = p_init[2]
    q1 = p_init[3]
    p, a0_st, a1_st = run_game_fala(agent_0_init_strategy=[[1-p0, p0], [1-p1, p1]], agent_1_init_strategy=[[1-q0, q0], [1-q1, q1]], s_0=0)
    abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results"))
    csv_file_name = "/pl_%.2f_%.2f_%.2f_%.2f_fala_strategy_trace.csv" % (p0, q0, p1, q1)
    file_name = abs_path + csv_file_name
    d_pd = pd.DataFrame(p)
    d_pd.to_csv(file_name, index=None)


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
        m = np.array([[average_payoff[s_i][0][0], average_payoff[s_i][0][1]],
                      [average_payoff[s_i][0][2], average_payoff[s_i][0][3]]])
    else:
        m = np.array([[average_payoff[s_i][1][0], average_payoff[s_i][1][2]],
                      [average_payoff[s_i][1][1], average_payoff[s_i][1][3]]])
    return m


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
        else:
            p_o = np.dot(p_m, [[q1], [1 - q1]])
            q_o = np.dot(q_m, [[p1], [1 - p1]])
            v_s = np.sum(v[4:8])
            dp = (np.dot([1, 0], p_o)[0] - np.dot([p1, 1-p1], p_o)[0]) * p1 * v_s
            dq = (np.dot([1, 0], q_o)[0] - np.dot([q1, 1-q1], q_o)[0]) * q1 * v_s
            p1 = valid_s(p1 + dp * step_size)
            q1 = valid_s(q1 + dq * step_size)
    return p0, q0, p1, q1


def run_task(p_init):
    t = np.arange(0, 10e5)
    step_size = 0.001
    s_n = 2
    print(p_init)
    for p_1, p_2 in [[0.9, 0.1], [0.5, 0.5]]:
        p0 = p_init[0]
        q0 = p_init[1]
        p1 = p_init[2]
        q1 = p_init[3]
        print(p_1, p_2)
        qvec = [p_1, p_2, p_2, p_2, p_1, p_2, p_2, p_2]
        f_p = np.array([3, 1, 4, 2, 3, 1, 4, 2])
        # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
        f_p = f_p.reshape(f_p.size, 1).transpose()
        f_q = np.array([3, 4, 1, 2, 3, 4, 1, 2])
        # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
        f_q = f_q.reshape(f_q.size, 1).transpose()
        d = []
        d.append([p0, q0, p1, q1])
        for _ in t:
            pl = [p0, p0, p0, p0, p1, p1, p1, p1]
            ql = [q0, q0, q0, q0, q1, q1, q1, q1]
            v, average_payoff = average_game(s_n, qvec, pl, ql, f_p, f_q)
            p0, q0, p1, q1 = evolve(s_n, average_payoff, p0, q0, p1, q1, step_size, v)
            d.append([p0, q0, p1, q1])
        abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results_random_initialization"))
        csv_file_name = "/p1_%.1f_p2_%.1f_pl_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (
            p_1, p_2, p_init[0], p_init[1], p_init[2], p_init[3])
        file_name = abs_path + csv_file_name
        d_pd = pd.DataFrame(d)
        d_pd.to_csv(file_name, index=None)


if __name__ == '__main__':
    # p, agent_0_strategy_trace, agent_1_strategy_trace = \
    #     run_game_fala(agent_0_init_strategy=[[0.5, 0.5], [0.5, 0.5]], agent_1_init_strategy=[[0.5, 0.5], [0.5, 0.5]], s_0=0)
    # print(p)
    # print(agent_0_strategy_trace[-1], agent_1_strategy_trace[-1])
    p = Pool()
    for p_init_time in range(10):
        p_sub = np.round(np.arange(0.1, 1.0, 0.1), 2)
        p_init = np.random.choice(p_sub, 4)
        p.apply_async(run_task_fala, args=(p_init,))
    p.close()
    p.join()
    print("All subprocesses done")
