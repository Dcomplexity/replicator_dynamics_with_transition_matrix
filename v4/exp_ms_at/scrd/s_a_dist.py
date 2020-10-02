import numpy as np
from scipy.linalg import solve
from game_env import *


# This is for the section 3.1 Average reward game

# q_s_z means the calculation of  Q^i(s_ap)
def q_sap_z(s_ap, z, a_l, pi, transition_matrix):
    q = 0
    for a_x in a_l:
        for a_y in a_l:
            q += transition_prob(s_ap, z, a_x, a_y, transition_matrix) * pi[0][s_ap][a_x] * pi[1][s_ap][a_y]
    return q


# The element if for Q^{i}, the index is for s_ap to z
def q_matrix(s_l, a_l, pi, transition_matrix):
    q_m = []
    for s in s_l:
        q_m.append([])
        for z in s_l:
            q_m[s].append(q_sap_z(s, z, a_l, pi, transition_matrix))
    return q_m


def gen_s_a_dist(s_l, a_l, s, a_x, a_y, pi, transition_matrix):
    q_m = q_matrix(s_l, a_l, pi, transition_matrix)
    if s == 0:
        # a = np.array([[transition_prob(s, s, a_x, a_y) - 1, q_m[1-s][s]], [transition_prob(s, 1-s, a_x, a_y), q_m[1-s][1-s] - 1]])
        # a = np.array([[transition_prob(s, s, a_x, a_y) - 1, q_m[1-s][s]], [1, 1]])
        a = np.array([[transition_prob(0, 1, a_x, a_y, transition_matrix), q_m[1][1] - 1, q_m[2][1]],
                      [transition_prob(0, 2, a_x, a_y, transition_matrix), q_m[1][2], q_m[2][2] - 1],
                      [1, 1, 1]])
    elif s == 1:
        # a = np.array([[q_m[1-s][1-s] - 1, transition_prob(s, 1-s, a_x, a_y)], [q_m[1-s][s], transition_prob(s, s, a_x, a_y)]])
        # a = np.array([[q_m[1-s][1-s] - 1, transition_prob(s, 1-s, a_x, a_y)], [1, 1]])
        a = np.array([[q_m[0][0] - 1, transition_prob(1, 0, a_x, a_y, transition_matrix), q_m[2][0]],
                      [q_m[0][2], transition_prob(1, 2, a_x, a_y, transition_matrix), q_m[2][2] - 1],
                      [1, 1, 1]])
    elif s == 2:
        a = np.array([[q_m[0][0] - 1, q_m[1][0], transition_prob(2, 0, a_x, a_y, transition_matrix)],
                      [q_m[0][1], q_m[1][1] - 1, transition_prob(2, 1, a_x, a_y, transition_matrix)],
                      [1, 1, 1]])

    b = np.array([0, 0, 1])
    x = solve(a, b)
    # print(s, a_x, a_y, x)
    # print(x)
    # x = x / np.sum(x)
    return x


def gen_payoff_pair(pds, a_l, s, a_x, a_y, pi, s_d):
    """
    :param pds: type of prisoners' dilemma
    :param a_l: action list
    :param s: current state
    :param a_x: action of player x
    :param a_y: action of player y
    :param pi: policy
    :param s_d: state distribution
    :return:
    """
    p = np.array(pds[s](a_x, a_y)) * s_d[s * 4 + a_x * 2 + a_y][s]
    for a_x_ in a_l:
        for a_y_ in a_l:
            # if a_x_ == a_y_ == 1:
            #     p += np.array(pds[1 - s](a_x_, a_y_)) * pi[0][1 - s][a_x_] * pi[1][1 - s][a_y_] * \
            #      s_d[s * 4 + a_x * 2 + a_y][1 - s]
            p += np.array(pds[1 - s](a_x_, a_y_)) * pi[0][1 - s][a_x_] * pi[1][1 - s][a_y_] * \
                 s_d[s * 4 + a_x * 2 + a_y][1 - s]
    return p


def run(pi, transition_matrix):
    pd_games = [play_pd_game_1, play_pd_game_2, play_pd_game_3]
    states = [0, 1, 2]
    actions = [0, 1]
    state_dist = []
    for s in states:
        for a_x in actions:
            for a_y in actions:
                state_dist.append(gen_s_a_dist(states, actions, s, a_x, a_y, pi, transition_matrix))
    payoff_matrix = []
    for s in states:
        for a_x in actions:
            for a_y in actions:
                payoff_matrix.append(gen_payoff_pair(pd_games, actions, s, a_x, a_y, pi, state_dist))
    return state_dist, payoff_matrix


if __name__ == "__main__":
    # the first dict for player 0 and the second dict for player 1,
    # In the dict, 0 for state 0 and 1 for state 1
    # policy_pi = [{0: [0.3, 0.7], 1: [0.3, 0.7]}, {0: [0.8, 0.2], 1: [0.8, 0.2]}]
    # policy_pi = [{0: [1.0, 0.0], 1: [1.0, 0.0]}, {0: [1.0, 0.0], 1: [1.0, 0.0]}]
    policy_pi = [{0: [0.0, 1.0], 1: [0.0, 1.0]}, {0: [0.0, 1.0], 1: [0.0, 1.0]}]
    # policy_pi = [{0: [0.1, 0.9], 1: [0.1, 0.9]}, {0: [0.1, 0.9], 1: [0.1, 0.9]}]
    transition_matrix = [[0.1, 0.1, 0.1, 0.9], [0.9, 0.9, 0.9, 0.1]]
    state_dist, payoff_matrix = run(policy_pi, transition_matrix)
    print(state_dist)
    print(payoff_matrix)
