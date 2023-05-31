"""Create DSPR model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Tue 30 May 2023 05:30:10 PM CST
# ***
# ************************************************************************************/
#
import os
import torch
import torch.nn as nn
import torch.fft
from typing import Tuple, Optional

import todos

import pdb

def get_fft_frequencies(grid: Tuple[int, int, int]) -> torch.Tensor:
    """Returns FFT frequencies for a given grid.
    """
    freqs = []
    for res in grid:
        freqs.append(torch.fft.fftfreq(res, d=1 / res))
    # (Pdb) torch.fft.fftfreq(res, d=1 / res)
    # tensor([   0.,    1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,
    #           10.,   11.,   12.,   13.,   14.,   15.,   16.,   17.,   18.,   19.,
    #           20.,   21.,   22.,   23.,   24.,   25.,   26.,   27.,   28.,   29.,
    #           30.,   31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,
    #           40.,   41.,   42.,   43.,   44.,   45.,   46.,   47.,   48.,   49.,
    #           50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,
    #           60.,   61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,
    #           70.,   71.,   72.,   73.,   74.,   75.,   76.,   77.,   78.,   79.,
    #           80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,
    #           90.,   91.,   92.,   93.,   94.,   95.,   96.,   97.,   98.,   99.,
    #          100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,  109.,
    #          110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,
    #          120.,  121.,  122.,  123.,  124.,  125.,  126.,  127., -128., -127.,
    #         -126., -125., -124., -123., -122., -121., -120., -119., -118., -117.,
    #         -116., -115., -114., -113., -112., -111., -110., -109., -108., -107.,
    #         -106., -105., -104., -103., -102., -101., -100.,  -99.,  -98.,  -97.,
    #          -96.,  -95.,  -94.,  -93.,  -92.,  -91.,  -90.,  -89.,  -88.,  -87.,
    #          -86.,  -85.,  -84.,  -83.,  -82.,  -81.,  -80.,  -79.,  -78.,  -77.,
    #          -76.,  -75.,  -74.,  -73.,  -72.,  -71.,  -70.,  -69.,  -68.,  -67.,
    #          -66.,  -65.,  -64.,  -63.,  -62.,  -61.,  -60.,  -59.,  -58.,  -57.,
    #          -56.,  -55.,  -54.,  -53.,  -52.,  -51.,  -50.,  -49.,  -48.,  -47.,
    #          -46.,  -45.,  -44.,  -43.,  -42.,  -41.,  -40.,  -39.,  -38.,  -37.,
    #          -36.,  -35.,  -34.,  -33.,  -32.,  -31.,  -30.,  -29.,  -28.,  -27.,
    #          -26.,  -25.,  -24.,  -23.,  -22.,  -21.,  -20.,  -19.,  -18.,  -17.,
    #          -16.,  -15.,  -14.,  -13.,  -12.,  -11.,  -10.,   -9.,   -8.,   -7.,
    #           -6.,   -5.,   -4.,   -3.,   -2.,   -1.])

    freqs = torch.stack(torch.meshgrid(freqs, indexing="ij"), dim=-1)
    # freqs.size() -- [256, 256, 256, 3]
    # freqs.min() -- tensor(-128.), freqs.max() -- tensor(127.), freqs.mean() -- tensor(-0.5000)
    return freqs


def get_gaussian_smoothing(
    input: torch.Tensor, sigma: int = 5, res: Optional[int] = None
) -> torch.Tensor:
    """Returns a gaussian smoothing kernel
    """
    # sigma = 1
    # res = 256
    # input.size() -- [1, 256, 256, 256, 3]

    if res is None:
        res = input.shape[0]
    # input ------ u
    _vector = torch.sum(input**2, dim=-1) # [1, 256, 256, 256]
    _scalar = -2 * (sigma / res) ** 2 # -3.0517578125e-05

    # (Pdb) aabb.min() -- tensor(0.2231), aabb.max() -- tensor(1.), aabb.mean() -- tensor(0.6264)
    return torch.exp(_scalar * _vector) # [1, 256, 256, 256]


def point_rasterization(
    points: torch.Tensor, features: torch.Tensor, grid: Tuple[int, int, int]
) -> torch.Tensor:
    """Returns a scalar/vector field as a grid of specified dimension.
    Values of the field are given by features while positions by points.
    Trilinear interpolation is used to approximate values in grid points.
    ATTENTION: points must be in [0, 1], changing center is not supported

    Parameters
    ----------
    points : torch.Tensor
        batch of sample positions
    features : torch.Tensor
        batch of sample values
    grid : Tuple[int, int, int]
        field dimension (grid size)

    Returns
    -------
    torch.Tensor
        scalar/vector field of shape [batch, nfeatures, *grid]
    """
    # grid = (256, 256, 256)
    # points.size() -- [1, 85127, 3]
    # features.size() -- [1, 85127, 3]

    rdevice = points.device # running device

    batchsize = points.shape[0] # 1
    samplesize = points.shape[1] # 85127
    dim = points.shape[2] # 3
    featuresize = features.shape[2] # 3

    # x0, y0, z0
    voxelcount = torch.tensor(grid).to(rdevice) # tensor([256, 256, 256])
    # s0, s1, s2
    voxelsize = 1 / voxelcount # tensor([0.0039, 0.0039, 0.0039])
    eps = 1e-5

    # compute neighbor indices
    lower_index = torch.floor(points / voxelsize).int()
    lower_index = lower_index.remainder(voxelcount)
    outliers = torch.cat(
        [
            torch.argwhere(lower_index > voxelcount - eps),
            torch.argwhere(lower_index < -eps),
        ]
    )
    if len(outliers) > 0:
        raise ValueError(f"{len(outliers)} points are outside grid limits")

    upper_index = torch.ceil(points / voxelsize).int()
    upper_index = upper_index.remainder(voxelcount)
    outliers = torch.cat(
        [
            torch.argwhere(upper_index > voxelcount + 1 - eps),
            torch.argwhere(upper_index < -eps),
        ]
    )
    if len(outliers) > 0:
        raise ValueError(f"{len(outliers)} points are outside grid limits")
    upper_index = upper_index.remainder(voxelcount)
    sample_index = torch.stack([lower_index, upper_index], dim=0) # [2, 1, 85127, 3]


    # ind0 = torch.floor(pts / voxelsize).int()  # (batch, num_points, dim)
    # ind1 = torch.fmod(torch.ceil(pts / voxelsize), voxelcount).int() # periodic wrap-around
    # ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)


    # all subsets of 2**dim elements
    # e.g. if dim=3 : 000, 001, 010, ..., 110, 111
    combinations = torch.stack(
        torch.meshgrid(*([torch.tensor([0, 1])] * dim), indexing="ij"), dim=-1
    ).reshape(2**dim, dim).to(rdevice)
    # combinations = torch.tensor([
    #     [0, 0, 0],
    #     [0, 0, 1],
    #     [0, 1, 0],
    #     [0, 1, 1],
    #     [1, 0, 0],
    #     [1, 0, 1],
    #     [1, 1, 0],
    #     [1, 1, 1]])

    # [0,1,..,dim-1] * (2**dim)
    # e.g if dim=3 : [[1,2,3],[1,2,3],...]
    selection = torch.arange(dim).repeat(2**dim, 1).to(rdevice)
    # selection = torch.tensor([
    #     [0, 1, 2],
    #     [0, 1, 2],
    #     [0, 1, 2],
    #     [0, 1, 2],
    #     [0, 1, 2],
    #     [0, 1, 2],
    #     [0, 1, 2],
    #     [0, 1, 2]])

    # creates all possible indexing combinations
    # e.g. (low, low, low), (low, low, up), ..., (up, up, up)
    # combinations generates all possible combinations:
    # (xl, xl, xl), (xl, xl, xu), ...
    # (yl, yl, yl), (yl, yl, yu), ...
    # (zl, zl, zl), (zl, zl, zu), ...
    # selection then selects the diagonal of each:
    # -> (xl, yl, zl), (xl, yl, zu), (xl, yu, zl), ...
    neighbor_index = sample_index[combinations, ..., selection] # [8, 3, 1, 85127]
    sample_index = neighbor_index.permute(2, 3, 0, 1)  # [1, 85127, 8, 3]， [batch, npoints, 2**dim, dim]

    # similarly we construct the cube coordinates
    lower_coords = lower_index * voxelsize
    upper_coords = (lower_index + 1) * voxelsize # voxelsize -- tensor([0.0039, 0.0039, 0.0039])
    sample_coords = torch.stack([lower_coords, upper_coords], dim=0) # [2, 1, 85127, 3]

    # here we invert the order from upper to lower instead
    neighbor_coords = sample_coords[1 - combinations, ..., selection]
    neighbor_coords = neighbor_coords.permute(2, 3, 0, 1) # neighbor_coords.size() -- [1, 85127, 8, 3

    # distances from points to neighbors
    neighbor_dist = torch.abs(points.unsqueeze(-2) - neighbor_coords)
    # scale distances to use cubes as a metric
    neighbor_dist /= voxelsize # neighbor_dist.size() -- [1, 85127, 8, 3]

    # compute trilinear weights for all 8 neighbors of each sample as:
    # n0 = |x_{n0} - x|/sx * |y_{n0} - y|/sy * |z_{n0} - z|/sz * (feature_value)
    weights = torch.prod(neighbor_dist, dim=-1, keepdim=False) # weights.size() -- [1, 85127, 8]

    # weights.unsqueeze(-1).size() -- [1, 85127, 8, 1]
    # features.unsqueeze(-2).size() -- [1, 85127, 1, 3]
    field_values = weights.unsqueeze(-1) * features.unsqueeze(-2) # [1, 85127, 8, 3]

    # initialize batch and feature indices
    batch_index = torch.arange(batchsize).expand(samplesize, 2**dim, batchsize) # [85127, 8, 1]
    batch_index = batch_index.permute(2, 0, 1) # [1, 85127, 8]
    feature_index = torch.arange(featuresize).reshape(1, 1, 1, featuresize, 1) # [1, 1, 1, 3, 1]

    # solve broadcasting issues
    batch_index = batch_index.unsqueeze(-1).unsqueeze(-1)
    batch_index = batch_index.expand(batchsize, samplesize, 2**dim, featuresize, 1) # [1, 85127, 8, 3, 1]
    sample_index = sample_index.unsqueeze(-2)

    sample_index = sample_index.expand(
        batchsize, samplesize, 2**dim, featuresize, dim
    ) # [1, 85127, 8, 3, 3]

    feature_index = feature_index.expand(
        batchsize, samplesize, 2**dim, featuresize, 1
    ) # [1, 85127, 8, 3, 1]

    # construct final index
    # [1, 85127, 8, 3, 1], [1, 85127, 8, 3, 1], [1, 85127, 8, 3, 3] ==> [1, 85127, 8, 3, 5]
    index = torch.cat([batch_index.to(rdevice), feature_index.to(rdevice), sample_index], dim=-1)

    # flatten all dimensions
    index = index.reshape(-1, dim + 2) # [2043048, 5]
    field_values = field_values.reshape(-1) # [2043048]

    # construct output grid
    output_size = torch.Size((batchsize, featuresize, *grid)) # torch.Size([1, 3, 256, 256, 256])
    output_grid = torch.zeros(output_size, dtype=field_values.dtype).view(-1).to(rdevice) # [50331648]

    # flatten the index
    # [1] + list(output_size[:0:-1]) -- [1, 256, 256, 256, 3]
    index_folds = torch.tensor([1] + list(output_size[:0:-1])).cumprod(0).flip(0).to(rdevice)
    # index_folds -- tensor([50331648, 16777216,    65536,      256,        1])
    index_flat = torch.sum(index * index_folds, dim=-1)

    # write field values to grid at index position
    # index_flat.size() -- [2043048]

    output_grid.scatter_add_(0, index_flat, field_values)
    output_grid = output_grid.view(*output_size)

    # output_grid.size() -- [1, 3, 256, 256, 256]
    return output_grid


def grid_interpolation(
    field_values: torch.Tensor, query_points: torch.Tensor
) -> torch.Tensor:
    """Given a scalar/vector field approximated using a grid and a set
    of query points returns field values for each query point approximated
    using trilinear interpolation.
    ATTENTION: query_points must be in [0, 1]

    Parameters
    ----------
    field_values : torch.Tensor
        a batch of field grid with shape = [batch, *grid, nfeatures]
    query_points : torch.Tensor
        a batch of query points with shape = [batch, npoints, dim]

    Returns
    -------
    torch.Tensor
        field values at query points with shape = [batch, npoints, nfeatures]
    """

    # field_values.size() -- [1, 256, 256, 256, 3]
    # query_points.size() -- [1, 85127, 3]

    batchsize = query_points.shape[0] # 1
    samplesize = query_points.shape[1] # 85127
    dim = query_points.shape[2] # 3
    voxelcount = torch.tensor(field_values.shape[1:-1]).to(field_values.device) # tensor([256, 256, 256])
    # s0, s1, s2
    voxelsize = 1 / voxelcount
    eps = 1e-5

    # compute neighbor indices
    lower_index = torch.floor(query_points / voxelsize).int()
    lower_index = lower_index.remainder(voxelcount)
    outliers = torch.cat(
        [
            torch.argwhere(lower_index > voxelcount - eps),
            torch.argwhere(lower_index < -eps),
        ]
    )
    if len(outliers) > 0:
        raise ValueError(f"{len(outliers)} points are outside grid limits")

    upper_index = torch.ceil(query_points / voxelsize).int()
    upper_index = upper_index.remainder(voxelcount)
    outliers = torch.cat(
        [
            torch.argwhere(upper_index > voxelcount + 1 - eps),
            torch.argwhere(upper_index < -eps),
        ]
    )
    if len(outliers) > 0:
        raise ValueError(f"{len(outliers)} points are outside grid limits")

    sample_index = torch.stack([lower_index, upper_index], dim=0) # [2, 85127, 8, 3]

    # all subsets of 2**dim elements
    # e.g. if dim=3 : 000, 001, 010, ..., 110, 111
    combinations = torch.stack(
        torch.meshgrid(*([torch.tensor([0, 1])] * dim), indexing="ij"), dim=-1
    ).reshape(2**dim, dim).to(field_values.device)
    # (Pdb) combinations
    # tensor([[0, 0, 0],
    #         [0, 0, 1],
    #         [0, 1, 0],
    #         [0, 1, 1],
    #         [1, 0, 0],
    #         [1, 0, 1],
    #         [1, 1, 0],
    #         [1, 1, 1]])

    # [0,1,..,dim-1] * (2**dim)
    # e.g if dim=3 : [[1,2,3],[1,2,3],...]
    selection = torch.arange(dim).repeat(2**dim, 1).to(field_values.device)
    # (Pdb) selection
    # tensor([[0, 1, 2],
    #         [0, 1, 2],
    #         [0, 1, 2],
    #         [0, 1, 2],
    #         [0, 1, 2],
    #         [0, 1, 2],
    #         [0, 1, 2],
    #         [0, 1, 2]])

    # creates all possible indexing combinations
    # e.g. (low, low, low), (low, low, up), ..., (up, up, up)
    # combinations generates all possible combinations:
    # (xl, xl, xl), (xl, xl, xu), ...
    # (yl, yl, yl), (yl, yl, yu), ...
    # (zl, zl, zl), (zl, zl, zu), ...
    # selection then selects the diagonal of each:
    # -> (xl, yl, zl), (xl, yl, zu), (xl, yu, zl), ...
    # sample_index.size() -- [2, 1, 85127, 3]
    neighbor_index = sample_index[combinations, ..., selection]
    #  neighbor_index.size() -- [8, 3, 1, 85127]
    sample_index = neighbor_index.permute(2, 3, 0, 1)  # [1, 85127, 8, 3], [batch, npoints, 2**dim, dim]
    # similarly we construct the cube coordinates
    lower_coords = lower_index * voxelsize
    upper_coords = (lower_index + 1) * voxelsize
    sample_coords = torch.stack([lower_coords, upper_coords], dim=0)

    # here we invert the order from upper to lower instead
    neighbor_coords = sample_coords[1 - combinations, ..., selection]
    neighbor_coords = neighbor_coords.permute(2, 3, 0, 1)

    # distances from points to neighbors
    neighbor_dist = torch.abs(query_points.unsqueeze(-2) - neighbor_coords)
    # scale distances to use cubes as a metric
    neighbor_dist /= voxelsize

    # compute trilinear weights for all 8 (if dim=3) neighbors of each sample as:
    # n0 = |x_{n0} - x|/sx * |y_{n0} - y|/sy * |z_{n0} - z|/sz * (feature_value)
    weights = torch.prod(neighbor_dist, dim=-1, keepdim=False) # [1, 85127, 8]

    # initialize batch and feature indices
    batch_index = torch.arange(batchsize).expand(samplesize, 2**dim, batchsize)
    batch_index = batch_index.permute(2, 0, 1)

    # get neighbor values
    # NOTE: if python version is 3.11 or higher this if can be removed and
    # value unpacking in subscripts was added
    if dim == 2:
        x, y = tuple(sample_index[..., i] for i in range(dim))
        neighbor_values = field_values[batch_index, x, y]
    elif dim == 3:
        x, y, z = tuple(sample_index[..., i] for i in range(dim))
        neighbor_values = field_values[batch_index, x, y, z]
    else:
        raise NotImplementedError("dim > 3 until transition to python 3.11")
    # neighbor_values.size() -- [1, 85127, 8, 3]

    # weighted sum over neighbor values
    # weights.unsqueeze(-1).size() -- [1, 85127, 8, 1]
    query_values = torch.sum(weights.unsqueeze(-1) * neighbor_values, dim=-2)

    return query_values # [1, 85127, 3]


class DPSR(nn.Module):
    def __init__(self, res=(256, 256, 256), sigma=2):
        super(DPSR, self).__init__()
        self.grid = res
        self.dim = len(res) # 3
        self.eps = 1e-6
        self.sigma = sigma

        # compute vector u
        u = get_fft_frequencies(self.grid).unsqueeze(0)  # [batch, *grid, dim]
        # pp u.size() -- [1, 256, 256, 256, 3]
        self.register_buffer("u", u)

        # compute vector g^~_{\sigma,r}(u)
        g = get_gaussian_smoothing(self.u, self.sigma, self.grid[0])  # [batch, *grid]
        # pp g.size() -- [1, 256, 256, 256]
        self.register_buffer("g", g)

    def forward(self, points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        # (Pdb) points.size() -- [1, 71265, 3]
        # (Pdb) normals.size() -- [1, 71265, 3]

        # compute vector v
        v = point_rasterization(points, normals, self.grid)  # [batch, dim, *grid]
        # v.size() -- [1, 3, 256, 256, 256]

        # compute vector v^~
        v_tilde = torch.fft.fftn(v, dim=(2, 3, 4)) # [1, 3, 256, 256, 256]
        v_tilde = v_tilde.permute([0, 2, 3, 4, 1])  # [batch, *grid, dim]
        # v_tilde.size() -- [1, 256, 256, 256, 3]

        # # compute vector u
        # u = get_fft_frequencies(self.grid).unsqueeze(0)  # [batch, *grid, dim]
        # # pp u.size() -- [1, 256, 256, 256, 3]

        # # compute vector g^~_{\sigma,r}(u)
        # g = get_gaussian_smoothing(u, self.sigma, self.grid[0])  # [batch, *grid]
        # # pp g.size() -- [1, 256, 256, 256]

        # compute scalar -i/2pi
        _scalar = -1j / (2 * torch.pi)
        # (Pdb) _scalar.real -- -0.0, _scalar.imag -- -0.15915494309189535

        # compute vector u @ v^~ / |u|^2
        _vector = torch.sum(self.u * v_tilde, dim=-1)  # dot-product  # [batch, *grid]
        _vector /= torch.sum(self.u**2, dim=-1) + self.eps  # [batch, *grid]
        # _vector.size() -- [1, 256, 256, 256]

        # compute vector \chi^~
        chi_tilde = self.g * (_scalar * _vector)  # [batch, *grid]
        # chi_tilde.size() -- [1, 256, 256, 256]

        # compute vector \chi'
        chi_prime = torch.fft.ifftn(chi_tilde, dim=(1, 2, 3))  # [batch, *grid]
        # =====================================================================
        chi_prime = chi_prime.real  # imag is zero
        # chi_prime.size() -- [1, 256, 256, 256]

        # compute vector \chi' restricted to x=c
        chi_c = grid_interpolation(chi_prime.unsqueeze(-1), points).squeeze(-1)  # [batch, *grid]
        _vector = chi_prime - torch.mean(chi_c, dim=-1)

        # compute scalar \chi' restricted to x=0
        chi_0 = _vector[:, 0, 0, 0] #  _vector.size() -- [1, 256, 256, 256]

        _scalar = 0.5 / torch.abs(chi_0) # tensor([105.0442])

        # compute vector \chi
        chi = _scalar * _vector

        return chi # chi.size() -- [1, 256, 256, 256]

    def __repr__(self):
        return f'DPSR: grid = {self.grid}, sigma = {self.sigma}'


if __name__ == "__main__":
    device = torch.device("cuda:0")
    pc_file = "../points/002.ply"
    points, normals = todos.data.load_3dply(pc_file, device=device)

    model = DPSR().to(device)
    model.eval()

    with torch.no_grad():
        chi = model(points, normals)

    todos.data.export_3dmesh(chi, "/tmp/dc.obj")

    os.system("meshlab /tmp/dc.obj")
