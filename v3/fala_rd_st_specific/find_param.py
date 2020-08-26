import numpy as np

# f_p = np.array([7, 5, 8, 6, 8, 6, 9, 7])
# f_q = np.array([7, 8, 5, 6, 6, 7, 4, 5])

# f_p = np.array([3, 1, 4, 2, 4, 2, 5, 3])
# f_q = np.array([3, 4, 1, 2, 4, 5, 2, 3])

# f_p = np.array([3, 1, 4, 2, 6, 4, 7, 5])
# f_q = np.array([3, 4, 1, 2, 6, 7, 4, 5])

# f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
# f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])

# f_p = np.array([3, 0, 5, 1, 2, -1, 4, 0])
# f_q = np.array([3, 5, 0, 1, 4, 6, 1, 2])

# f_p = np.array([3, 2, 4, 1, 3.5, 2.5, 4, 1])
# f_q = np.array([3, 4, 2, 1, 2.5, 3.5, 2, 1])

# f_p = np.array([7, 5, 8, 6, 6, 4, 7, 5])
# f_q = np.array([7, 8, 5, 6, 8, 9, 6, 7])

# f_p = np.array([2, 1, 1, 3, 2.5, 1.5, 1.5, 3.5])
# f_q = np.array([3, 1, 1, 2, 2.5, 0.5, 0.5, 1.5])
f_p = np.array([3, 1, 4, 2, 8, 6, 9, 7])
f_q = np.array([3, 4, 1, 2, 8, 9, 6, 7])
print(f_p, f_q)

min_test = -20
max_test = 20

test_phi = np.arange(min_test, max_test)
test_phi = test_phi[test_phi != 0]
test_phi = 1 / test_phi
test_kai_1 = np.arange(min_test, max_test)
test_kai_2 = test_kai_1[test_kai_1 != 0]
test_kai_2 = 1 / test_kai_2
test_kai = np.append(test_kai_1, test_kai_2)
test_v1 = np.arange(min_test, max_test)
test_v2 = np.arange(min_test, max_test)

for phi in test_phi:
    print(phi)
    for kai in test_kai:
        # print(kai)
        for v1 in test_v1:
            # print(v1)
            for v2 in test_v2:
                # print(v2)
                t0 = phi * ((f_p[0] - v1) - kai * (f_q[0] - v2)) + 1
                t1 = phi * ((f_p[1] - v1) - kai * (f_q[1] - v2)) + 1
                t2 = phi * ((f_p[2] - v1) - kai * (f_q[2] - v2)) + 1
                t3 = phi * ((f_p[3] - v1) - kai * (f_q[3] - v2)) + 1
                t4 = phi * ((f_p[4] - v1) - kai * (f_q[4] - v2))
                t5 = phi * ((f_p[5] - v1) - kai * (f_q[5] - v2))
                t6 = phi * ((f_p[6] - v1) - kai * (f_q[6] - v2))
                t7 = phi * ((f_p[7] - v1) - kai * (f_q[7] - v2))
                t_list = [t0, t1, t2, t3, t4, t5, t6, t7]
                flag = 0
                for value in t_list:
                    if value >= 1 or value <= 0:
                        flag = 1
                        break
                if flag == 0:
                    # if kai > 0:
                    print([phi, kai, v1, v2])

