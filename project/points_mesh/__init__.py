"""point cloud to mesh Package."""  # coding=utf-8
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
from . import mesh
import pdb


def get_tvm_model():
    """
    TVM model base on torch.jit.trace, that's why we construct it from scratch
    """

    model = mesh.MeshSDF()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running tvm model model on {device} ...")

    return model, device


def get_torch_model():
    """Create model."""
    model = mesh.MeshSDF()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")
    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/points_mesh.torch"):
    #     model.save("output/points_mesh.torch")

    return model, device


def mesh_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_torch_model()

    # load files
    pc_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(pc_filenames))
    for filename in pc_filenames:
        progress_bar.update(1)

        points, normals = todos.data.load_3dply(filename, device=device)
        # ignore normals
        with torch.no_grad():
            chi, pred_points, pred_normals = model(points)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        todos.data.save_3dply(pred_points, pred_normals, output_file)

        output_file = output_file.split('.')[0] + ".obj" # replace "*.ply to *.obj"
        todos.data.export_3dmesh(chi, output_file)

    todos.model.reset_device()
