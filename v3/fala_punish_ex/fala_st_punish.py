import numpy as np
import random
import pandas as pd
import os

from copy import deepcopy

pd_game = [[7, 7], [9, 6], [6, 9], [8, 8]]

# if state = 0, punish, elif state = 1, not punish
# punish_prob = [8/20, 19/20, 1/20, 12/20, 8/20, 19/20, 1/20, 12/20]
punish_prob = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
# punish_prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


def play_pd_game(a_x, a_y):
    return pd_game[a_x * 2 + a_y]


def punish_state(s, a_x, a_y):
    prob = punish_prob[s * 4 + a_x * 2 + a_y]
    if random.random() < prob:
        s_ = 0
    else:
        s_ = 1
    return s_


class Agent:
    def __init__(self, alpha, agent_id):
        self.id = agent_id
        self.strategy = []
        self.actions = [0, 1]  # 0 for defection and 1 for cooperation
        # self.states = [0, 1]  # there are two state: 0 and 1
        self.len_action = len(self.actions)
        self.alpha = alpha
        self.strategy_trace = []

    def get_strategy(self):
        return self.strategy

    def initial_strategy(self):
        self.strategy = [1 / self.len_action for _ in range(self.len_action)]

    def set_strategy(self, new_stra=None):
        """
        set a specified strategy
        :param new_s: the probability to play cooperation (action[1])
        :return:
        """
        if new_stra:
            self.strategy = new_stra

    def choose_action(self):
        a = np.random.choice(np.array(self.actions), size=1, p=self.strategy)[0]
        return a

    def update_strategy(self, a, r):
        for action in self.actions:
            if action == a:
                self.strategy[action] = self.strategy[action] + self.alpha * r * (1 - self.strategy[action])
            else:
                self.strategy[action] = self.strategy[action] - self.alpha * r * self.strategy[action]

    def record_strategy(self):
        self.strategy_trace.append(deepcopy(self.strategy))


def run_game_fala(agent_x_init_strategy, agent_y_init_strategy, s_0):
    agent_x = Agent(alpha=0.0001, agent_id=0)
    agent_y = Agent(alpha=0.0001, agent_id=1)
    agent_x.initial_strategy()
    agent_y.initial_strategy()
    agent_x.set_strategy(agent_x_init_strategy)
    agent_y.set_strategy(agent_y_init_strategy)
    cur_s = s_0
    for _ in range(int(10e4)):
        print(_)
        agent_x.record_strategy()
        agent_y.record_strategy()
        a_x = agent_x.choose_action()
        a_y = agent_y.choose_action()
        r_x, r_y = play_pd_game(a_x, a_y)
        print('cur_s', cur_s)
        if cur_s == 0:
            r_x = r_x - 5
            r_y = r_y - 5
        print(r_x, r_y)
        agent_x.update_strategy(a_x, r_x)
        agent_y.update_strategy(a_y, r_y)
        cur_s = punish_state(cur_s, a_x, a_y)

    p = []
    for i in range(len(agent_x.strategy_trace)):
        p.append([agent_x.strategy_trace[i][1], agent_y.strategy_trace[i][1]])
    return p, agent_x.strategy_trace, agent_y.strategy_trace


def run_task_fala(s_init):
    p_init = s_init[0]
    q_init = s_init[1]
    p, x_st, y_st = run_game_fala(agent_x_init_strategy=[1 - p_init, p_init],
                                  agent_y_init_strategy=[1 - q_init, q_init], s_0=1)
    abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results_st_punish"))
    csv_file_name = "fala_%.2f_%.2f_strategy_trace.csv" %(p_init, q_init)
    file_name = abs_path + csv_file_name
    d_pd = pd.DataFrame(p)
    d_pd.to_csv(file_name, index=None)

if __name__ == '__main__':
    run_task_fala(s_init=[0.4, 0.4])
