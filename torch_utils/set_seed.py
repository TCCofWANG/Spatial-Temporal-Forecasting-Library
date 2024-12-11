import torch
import random
import os
import numpy as np

# 测试过，这种设置是可以重复实验的
def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False  # train speed is slower after enabling this opts.
    # # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    #
    # # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    # torch.use_deterministic_algorithms(True)






