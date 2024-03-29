from game_env import *
import s_a_dist as sad
import s_pi_dist as spd
import os
import pandas as pd
import datetime
import numpy as np
import argparse

from multiprocessing import Pool


def valid_s(s_value):
    if s_value < 0.001:
        s_new = 0.001
    elif s_value > 0.999:
        s_new = 0.999
    else:
        s_new = s_value
    return s_new


def calc_payoff(agent_id, s, a_l, mixed_s, p_m, pi):
    p = 0
    # if agent_id == 0:
    #     for act_i in a_l:
    #         p_j = 0
    #         for act_j in a_l:
    #             p_j += p_m[s * 4 + act_i * 2 + act_j][0] * pi[1 - agent_id][s][act_j]
    #         p += mixed_s[act_i] * p_j
    # else:
    #     for act_i in a_l:
    #         p_j = 0
    #         for act_j in a_l:
    #             p_j += p_m[s * 4 + act_j * 2 + act_i][1] * pi[1 - agent_id][s][act_j]
    #         p += mixed_s[act_i] * p_j
    if agent_id == 0:
        for act_i in a_l:
            p_i = 0
            for act_j in a_l:
                p_i += p_m[s * 4 + act_i * 2 + act_j][0] * pi[1 - agent_id][s][act_j]
            p += mixed_s[act_i] * p_i
    else:
        for act_j in a_l:
            p_j = 0
            for act_i in a_l:
                p_j += p_m[s * 4 + act_i * 2 + act_j][1] * pi[1 - agent_id][s][act_i]
            p += mixed_s[act_j] * p_j
    return p


def evolve(strategy, step_size, transition_matrix):
    s_l = [0, 1, 2]
    a_l = [0, 1]
    # The initial strategies are set here.
    # s00 for p0, s10 for q0, s01 for p1, s11 for q1, s02 for p2, s12 for q2
    s00, s10, s01, s11, s02, s12 = strategy
    # s00 = strategy[0]
    # s01 = strategy[2]
    # s10 = strategy[1]
    # s11 = strategy[3]
    pi = [{0: [1 - s00, s00], 1: [1 - s01, s01], 2: [1 - s02, s02]},
          {0: [1 - s10, s10], 1: [1 - s11, s11], 2: [1 - s12, s12]}]
    s_dist, p_matrix = sad.run(pi, transition_matrix)
    s_pi_dist = spd.gen_s_pi_dist(s_l, a_l, pi, transition_matrix)
    # agent 0 in state 0
    ds00 = (calc_payoff(0, 0, a_l, [0, 1], p_matrix, pi) - calc_payoff(0, 0, a_l, pi[0][0], p_matrix, pi)) * s00 * \
           s_pi_dist[0]
    # agent 0 in state 1
    ds01 = (calc_payoff(0, 1, a_l, [0, 1], p_matrix, pi) - calc_payoff(0, 1, a_l, pi[0][1], p_matrix, pi)) * s01 * \
           s_pi_dist[1]
    # agent 0 in state 2
    ds02 = (calc_payoff(0, 2, a_l, [0, 1], p_matrix, pi) - calc_payoff(0, 2, a_l, pi[0][2], p_matrix, pi)) * s02 * \
           s_pi_dist[2]
    # agent 1 in state 0
    ds10 = (calc_payoff(1, 0, a_l, [0, 1], p_matrix, pi) - calc_payoff(1, 0, a_l, pi[1][0], p_matrix, pi)) * s10 * \
           s_pi_dist[0]
    # agent 1 in state 1
    ds11 = (calc_payoff(1, 1, a_l, [0, 1], p_matrix, pi) - calc_payoff(1, 1, a_l, pi[1][1], p_matrix, pi)) * s11 * \
           s_pi_dist[1]
    # agent 2 in state 2
    ds12 = (calc_payoff(1, 2, a_l, [0, 1], p_matrix, pi) - calc_payoff(1, 2, a_l, pi[1][2], p_matrix, pi)) * s12 * \
           s_pi_dist[2]
    s00 = valid_s(s00 + ds00 * step_size)
    s01 = valid_s(s01 + ds01 * step_size)
    s02 = valid_s(s02 + ds02 * step_size)
    s10 = valid_s(s10 + ds10 * step_size)
    s11 = valid_s(s11 + ds11 * step_size)
    s12 = valid_s(s12 + ds12 * step_size)
    return [s00, s10, s01, s11, s02, s12]


def run_task(p_init):
    states = [0, 1, 2]
    actions = [0, 1]
    t = np.arange(0, 10e5)
    # t = np.arange(0, 1000)
    step_length = 0.001
    print(p_init)
    for p_1, p_2 in [[0.9, 0.1]]:
        print(p_1, p_2)
        # transition_matrix = [[p_2, p_1, p_1, p_2], [p_2, p_1, p_1, p_2]]
        transition_matrix = [[0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.8, 0.1, 0.1],
                             [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.8, 0.1, 0.1],
                             [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.8, 0.1, 0.1]]
        p = p_init
        d = []
        d.append(p)
        for _ in t:
            if _ % 10000 == 0:
                print('scrd', _)
            p = evolve(p, step_length, transition_matrix)
            # if _ % 1000 == 0:
            #     print(p)
            d.append(p)
        abs_path = os.path.abspath(os.path.join(os.getcwd(), "../results_st_at"))
        csv_file_name = "/scrd_ms_st_at_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_strategy_trace.csv" % (
            p_init[0], p_init[1], p_init[2], p_init[3], p_init[4], p_init[5])
        file_name = abs_path + csv_file_name
        d_pd = pd.DataFrame(d)
        d_pd.to_csv(file_name, index=None)


def run_expected_payoff_matrix_task():
    s_l = [0, 1, 2]
    a_l = [0, 1]
    s00, s10, s01, s11, s02, s12 = [1, 1, 1, 1, 1, 1]
    transition_matrix = [[0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.9, 0.05, 0.05],
                         [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.9, 0.05, 0.05],
                         [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.9, 0.05, 0.05]]
    pi = [{0: [1 - s00, s00], 1: [1 - s01, s01], 2: [1 - s02, s02]},
            {0: [1 - s10, s10], 1: [1 - s11, s11], 2: [1 - s12, s12]}]
    s_dist, p_matrix = sad.run(pi, transition_matrix)
    s_pi_dist = spd.gen_s_pi_dist(s_l, a_l, pi, transition_matrix)
    print(calc_payoff(0, 0, a_l, [1, 0], p_matrix, pi))


def read_s_init():
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "ms_init_file.csv")
    data = pd.read_csv(f, usecols=['0', '1', '2', '3', '4', '5'])
    s_init = np.array(data).tolist()
    return s_init


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p1', '--p1', type=float, default=0.9, help="transition probability")
    # parser.add_argument('-p2', '--p2', type=float, default=0.1, help="transition probability")
    # args = parser.parse_args()

    # transition_matrix = [[0.1, 0.1, 0.1, 0.9], [0.9, 0.9, 0.9, 0.1]]
    # transition_matrix = [[0.9, 0.9, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]]
    # transition_matrix = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    # p_1 = args.p1
    # p_2 = args.p2
    # print(p_1, p_2)

    # s_init_list = read_s_init()
    # init_num = len(s_init_list)
    # p = Pool()
    # for _ in range(init_num):
    #     s_init = s_init_list[_][:]
    #     p.apply_async(run_task, args=(s_init,))
    # p.close()
    # p.join()
    # print("All subpocesses done.")

    # run_task(s_init_list[0])

    run_expected_payoff_matrix_task()