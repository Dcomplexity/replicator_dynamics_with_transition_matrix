import numpy as np
import pandas as pd
import os

abs_path = os.getcwd()
dir_name = os.path.join(abs_path)

f = os.path.join(dir_name, "p_init_file.csv")
data = pd.read_csv(f, usecols=['0', '1', '2', '3'])
p_init = np.array(data).tolist()
print(p_init)
