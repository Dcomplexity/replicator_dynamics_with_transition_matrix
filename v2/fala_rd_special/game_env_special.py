import numpy as np
import random

pd_game_1 = [[2, 2], [4, 1], [1, 4], [3, 3]]
# pd_game_2 = [[2, 2], [4, 1], [1, 4], [3, 3]]
pd_game_2 = [[6, 6], [8, 5], [5, 8], [7, 7]]
# pd_game_2 = [[7, 7], [9, 6], [6, 9], [8, 8]]
# pd_game_1 = [[2, 2], [10, 0], [0, 10], [3, 3]]
# pd_game_2 = [[1, 1], [10, 0], [0, 10], [4, 4]]

# This is a speical transition strategy, switch the strategy of cc and dd
transition_prob = [27/40, 39/40, 6/40, 18/40, 23/40, 35/40, 2/40, 14/40]
# transition_prob = [0.5, 0.3, 0.9, 0.7, 0.3, 0.1, 0.7, 0.5]
# transition_prob = [0.4, 0.2, 0.8, 0.6, 0.4, 0.2, 0.8, 0.6]
# transition_prob = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9]
# transition_prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# transition_prob = [0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1]
# transition_prob = [0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1]


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

