import numpy as np
import pandas as pd
import os
import math
from copy import deepcopy
from game_env_exploration import *
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
        print(new_stra)

    def choose_action(self, s):
        # print(self.id, self.strategy)
        a = np.random.choice(np.array(self.actions), size=1, p=self.strategy[s])[0]
        return a

    def update_strategy(self, s, a, r):
        for action in self.actions:
            if action == a:
                self.strategy[s][action] = valid_s(
                    self.strategy[s][action] + self.alpha * r * (1 - self.strategy[s][action]))
            else:
                self.strategy[s][action] = valid_s(self.strategy[s][action] - self.alpha * r * self.strategy[s][action])

    def record_strategy(self):
        self.strategy_trace.append(deepcopy(self.strategy))


def run_game_fala(agent_0_init_strategy, agent_1_init_strategy, s_0):
    agent_0 = Agent(alpha=0.000001, agent_id=0)
    agent_1 = Agent(alpha=0.000001, agent_id=1)
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
    for _ in range(20000000):
        if _ % 10000 == 0:
            print('fala', _)
        agent_0.record_strategy()
        agent_1.record_strategy()
        if visited[cur_s] == 0 or visited[1 - cur_s] == 0:
            if visited[cur_s] == 0:
                # print("1st")
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
                # print("2nd")
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
                r_sum_0[cur_s] = r_sum_0[cur_s] + r_0
                r_sum_1[cur_s] = r_sum_1[cur_s] + r_1
                time_step[cur_s] = 0
                time_step[cur_s] += 1
                cur_s = next_state(cur_s, a_0, a_1)
        else:
            # print('3rd')
            # print('round ', _)
            # print('r_sum_0 ', r_sum_0, 'r_sum_1 ', r_sum_1)
            # print('action_0 ', action_t_0, 'action_t_1 ', action_t_1)
            # print('strategy_0 ', agent_0.strategy, 'strategy_1 ', agent_1.strategy)
            tau_0[cur_s] = r_sum_0[cur_s] / time_step[cur_s]
            tau_1[cur_s] = r_sum_1[cur_s] / time_step[cur_s]
            # print('cur_s', cur_s)
            # print('time_step ', time_step)
            # print('tau_0 ', tau_0, 'tau_1 ', tau_1)
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
    p, a0_st, a1_st = run_game_fala(agent_0_init_strategy=[[1 - p0, p0], [1 - p1, p1]],
                                    agent_1_init_strategy=[[1 - q0, q0], [1 - q1, q1]], s_0=1)
    abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results"))
    csv_file_name = "/fala_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (p0, q0, p1, q1)
    file_name = abs_path + csv_file_name
    d_pd = pd.DataFrame(p)
    d_pd.to_csv(file_name, index=None)


def read_p_init():
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "p_init_file.csv")
    data = pd.read_csv(f, usecols=['0', '1', '2', '3'])
    p_init = np.array(data).tolist()
    return p_init


if __name__ == '__main__':
    p_fala = Pool()
    # p_sub = np.round(np.arange(0.1, 1.0, 0.1), 2)
    # p_init_list = []
    # for _ in range(10):
    #     p_init_list.append(np.random.choice(p_sub, 4))
    # p_init_pd = pd.DataFrame(p_init_list)
    # p_init_pd.to_csv('./p_init_file.csv')
    # print("save the init probability")
    p_init_list = read_p_init()
    init_num = len(p_init_list)
    for _ in range(10):
        p_init = p_init_list[_][:]
        p_fala.apply_async(run_task_fala, args=(p_init,))
    p_fala.close()
    p_fala.join()
    print("All subprocesses done")
