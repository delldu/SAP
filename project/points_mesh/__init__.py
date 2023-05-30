"""points to mesh Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 30 May 2023 05:30:10 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import todos
from . import guided

import pdb


def get_tvm_model():
    """
    TVM model base on torch.jit.trace, that's why we construct it from scratch
    """

    model = guided.DeepGuidedFilterAdvanced()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running tvm model model on {device} ...")

    return model, device


def get_torch_model():
    """Create model."""
    base = guided.DeepGuidedFilterAdvanced()
    base.load_weights(model_path="models/points_mesh.pth")
    model = todos.model.ResizePadModel(base)
    # model = todos.model.GridTileModel(base)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")
    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/points_mesh.torch"):
        model.save("output/points_mesh.torch")

    return model, device


def mesh_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_torch_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
