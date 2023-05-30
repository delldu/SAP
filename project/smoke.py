# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 30 May 2023 05:30:10 PM CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
import random
import torch

import points_mesh

from tqdm import tqdm

if __name__ == "__main__":
    model, device = points_mesh.get_torch_model()

    N = 100
    B, C, H, W = 1, 3, model.max_h, model.max_w

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        h = random.randint(-16, 16)
        w = random.randint(-16, 16)
        x = torch.randn(B, C, H + h, W + w)
        # print("x: ", x.size())

        start_time = time.time()
        with torch.no_grad():
            y = model(x.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")
