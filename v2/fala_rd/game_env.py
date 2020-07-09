import numpy as np
import random

pd_game_1 = [[2, 2], [4, 1], [1, 4], [3, 3]]
pd_game_2 = [[2, 2], [4, 1], [1, 4], [3, 3]]

transition_prob = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9]

def play_pd_game_1(a_x, a_y):
    return pd_game_1[a_x * 2 + a_y]


def play_pd_game_2(a_x, a_y):
    return pd_game_2[a_x * 2 + a_y]


def next_state(s, a_x, a_y):
    prob = transition_prob[s * 4 + a_x * 2 + a_y]
    if random.random() < prob:
        s_ = 0
    else:
        s_ = 1
    return s_

