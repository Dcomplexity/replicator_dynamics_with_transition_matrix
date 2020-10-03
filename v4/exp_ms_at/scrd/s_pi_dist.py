import numpy as np
from scipy.linalg import solve
from game_env import *
import random


def q_z_s(z, s, a_l, pi, transition_matrix):
    q = 0
    for a_x in a_l:
        for a_y in a_l:
            q += transition_prob(z, s, a_x, a_y, transition_matrix) * pi[0][z][a_x] * pi[1][z][a_y]
    return q


def q_matrix(s_l, a_l, pi, transition_matrix):
    q_m = []
    for z in s_l:
        q_m.append([])
        for s in s_l:
            q_m[z].append(q_z_s(z, s, a_l, pi, transition_matrix))
    return q_m


def gen_s_pi_dist(s_l, a_l, pi, transition_matrix):
    q_m = q_matrix(s_l, a_l, pi, transition_matrix)
    # a = np.array([[q_m[0][0] - 1, q_m[1][0]], [1, 1]]
    a = np.array([[q_m[0][0] - 1, q_m[1][0], q_m[2][0]], [q_m[0][1], q_m[1][1] - 1, q_m[2][1]], [1, 1, 1]])
    b = np.array([0, 0, 1])
    x = solve(a, b)
    # print(x)
    # x = x / np.sum(x)
    return x


if __name__ == '__main__':
    # the first dict for player 0 and the second dict for player 1,
    # In the dict, 0 for state 0 and 1 for state 1
    s00 = random.random()
    s01 = random.random()
    s02 = random.random()
    s10 = random.random()
    s11 = random.random()
    s12 = random.random()
    policy_pi = [{0: [1 - s00, s00], 1: [1 - s01, s01], 2: [1 - s02, s02]},
                 {0: [1 - s10, s10], 1: [1 - s11, s11], 2: [1 - s12, s12]}]
    # policy_pi = [{0: [0.1, 0.9], 1: [0.1, 0.9]}, {0: [0.1, 0.9], 1: [0.1, 0.9]}]
    # transition_matrix = [[0.1, 0.1, 0.1, 0.9], [0.9, 0.9, 0.9, 0.1]]
    transition_matrix = [[0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.8, 0.1, 0.1],
                         [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.8, 0.1, 0.1],
                         [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.1, 0.45, 0.45], [0.8, 0.1, 0.1]]
    states = [0, 1, 2]
    actions = [0, 1]
    state_dist = gen_s_pi_dist(states, actions, policy_pi, transition_matrix)
    print(state_dist)
