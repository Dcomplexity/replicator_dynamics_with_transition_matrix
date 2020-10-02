import numpy as np
import random
# b = 2.0
# c = 1.0
# pd_game_1 = [[0.2, 0.2], [1, 0], [0, 1], [0.3, 0.3]]
# pd_game_2 = [[0.1, 0.1], [1, 0], [0, 1], [0.4, 0.4]]
# pd_game_1 = [[0, 0], [1.2, -1], [-1, 1.2], [0.2, 0.2]]
# pd_game_1 = [[0, 0], [b, -c], [-c, b], [b-c, b-c]]
# # pd_game_2 = [[0, 0], [1.2, -1], [-1, 1.2], [0.2, 0.2]]
# # pd_game_2 = [[0, 0], [2, -1], [-1, 2], [1, 1]]
# pd_game_2 = [[0, 0], [b, -c], [-c, b], [b-c, b-c]]

# pd_game_1 = [[2, 2], [4, 1], [1, 4], [3, 3]]
# pd_game_2 = [[2, 2], [4, 1], [1, 4], [3, 3]]
pd_game_1 = [[2, 2], [10, 0], [0, 10], [3, 3]]
pd_game_2 = [[1, 1], [10, 0], [0, 10], [4, 4]]

# pd_game_1 = [[2, 2], [10, 0], [0, 10], [3, 3]]
# pd_game_2 = [[1, 1], [10, 0], [0, 10], [4, 4]]

def play_pd_game_1(a_x, a_y):
    return pd_game_1[a_x * 2 + a_y]

def play_pd_game_2(a_x, a_y):
    return pd_game_2[a_x * 2 + a_y]


def transition_prob(s, s_, a_x, a_y, transition_matrix):
    return transition_matrix[s * 4 + a_x * 2 + a_y][s_]

def next_state(s, a_x, a_y, transition_matrix):
    prob = transition_matrix[s * 4 + a_x * 2 + a_y]
    s_ = np.random.choice([0, 1], p=prob)
    return s_


if __name__ == "__main__":
    s_sum = 0
    # The probability to next state based on the joint action
    # transition_matrix = [[0.1, 0.8, 0.9, 0.1], [0.1, 0.9, 0.9, 0.1]]
    # transition_matrix = [[0.9, 0.9, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]]
    # transition_matrix = [[0.1, 0.1, 0.1, 0.9], [0.9, 0.9, 0.9, 0.1]]
    # transition_matrix = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    for i in range(1000):
        p_s = next_state(1, 1, 1, transition_matrix)
        s_sum += p_s
    print(s_sum)