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

import torch
import torch.nn.functional as F
from pytorch3d.io import IO, save_obj
from pytorch3d.ops.marching_cubes import marching_cubes

def minmax_scaling(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale all points to be within [0,1]
    """
    min = points.min() - eps
    max = points.max() + eps
    return (points - min) / (max - min)

def unitsphere_scaling(points: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Center input in origin and shift in [-1,1] * scale.
    """
    # scale = 0.9

    center = points.mean(dim=1)
    shifted = points - center
    scaleto1 = torch.abs(shifted).max()
    zoomed = shifted / (scale * scaleto1)

    return zoomed

def load_ply(filename, device=torch.device("cpu")):
    pc = IO().load_pointcloud(filename, device)

    PREPROC = lambda x: torch.sigmoid(unitsphere_scaling(x, scale=0.9))
    points = PREPROC(pc.points_list()[0].unsqueeze(0))
    normals = F.normalize(pc.normals_list()[0]).unsqueeze(0)

    return points, normals

def export_mesh(chi, filename, device=torch.device("cuda:0")):
    v, f = marching_cubes(chi.to(device), isolevel=0)
    save_obj(filename, v[0], f[0]);
