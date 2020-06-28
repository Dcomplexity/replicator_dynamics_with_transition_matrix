import numpy as np
from scipy.linalg import null_space
# print("hello world")
# p = [0.5, 0.5, 0.5, 0.5]
# q = [0.5, 0.5, 0.5, 0.5]
# sg_p = [p[:] for i in range(4)]
# sg_q = [q[:] for i in range(4)]
# print(sg_p, sg_q)
# # sg_p = np.array(sg_p)
# # sg_q = np.array(sg_q)
# for i in range(4):
#     sg_p[i][i] = 1
#     sg_q[i][i] = 1
# print(sg_p, sg_q)
# p = [0.8, 0.5, 0.5, 0.5]
# p = [0.0, 0.0, 0.0, 0.0]
p = [1.0, 1.0, 1.0, 1.0]
# p = [0.8, 0.8, 0.8, 0.8]

q = [0.5, 0.5, 0.5, 0.5]
f_p = np.array([3, 0, 5, 1])
f_q = np.array([3, 5, 0, 1])

m = np.array([[p[0] * q[0], p[0] * (1 - q[0]), (1 - p[0]) * q[0], (1 - p[0]) * (1 - q[0])],
              [p[1] * q[1], p[1] * (1 - q[1]), (1 - p[1]) * q[1], (1 - p[1]) * (1 - q[1])],
              [p[2] * q[2], p[2] * (1 - q[2]), (1 - p[2]) * q[2], (1 - p[2]) * (1 - q[2])],
              [p[3] * q[3], p[3] * (1 - q[3]), (1 - p[3]) * q[3], (1 - p[3]) * (1 - q[3])]])

null_matrix = np.transpose(m) - np.eye(4)
v = null_space(null_matrix)
v = v / np.sum(v)
f_p = f_p.reshape(f_p.size, 1).transpose()
f_q = f_q.reshape(f_q.size, 1).transpose()
r_p = np.dot(f_p, v)[0]
r_q = np.dot(f_q, v)[0]
v = v.flatten()
print(v)
print(r_p, r_q)

print("Hello World")