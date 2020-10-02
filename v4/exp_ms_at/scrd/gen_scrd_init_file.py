import os
import numpy as np
import pandas as pd

def read_s_init():
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "../ms_init_file.csv")
    data = pd.read_csv(f, usecols=['0', '1', '2', '3'])
    s_init = np.array(data).tolist()
    return s_init


if __name__ == '__main__':
    s_init = read_s_init()
    for item in s_init:
        temp = item[2]
        item[2] = item[1]
        item[1] = temp
    scrd_s_init = pd.DataFrame(s_init)
    abs_path = os.getcwd()
    dir_name = os.path.join(abs_path)
    f = os.path.join(dir_name, "scrd_ms_init_file.csv")
    scrd_s_init.to_csv(f)
