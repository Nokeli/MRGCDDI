# import pandas as pd
# import numpy as py
# import random
import os
import torch
import random

import copy


import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=False)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for k in range(5):

    # os.system('python main.py --out_file test.txt --zhongzi '+str(k))
    os.system(r'D:\Users\Li\miniconda3\envs\pytorch\python main.py --out_file test_withReadout+1+0.1+0.1+loss_rd0.3.txt --zhongzi ' + str(k))



