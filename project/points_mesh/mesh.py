"""Create model."""
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
from torch.nn import functional as F

from torch_scatter import scatter_mean, scatter_max
from . import unet3d
from . import dpsr
import pdb

def normalize_3d_coordinate(p):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
    '''   
    if p.max() >= 1:
        p[p >= 1] = 1 - 10e-6
    if p.min() < 0:
        p[p < 0] = 0.0
    return p

# xxxx8888
def coordinate3d_index(x, reso):
    x = (x * reso).long()
    index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

# xxxx8888
class map2local(object):
    def __init__(self, s):
        super().__init__()
        self.s = s # 'grid_resolution' -- 32

    def __call__(self, p):
        p = (p % self.s) / self.s
        p[p < 0] = 0.0
        return p

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        # xxxx8888
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        unet3d_kwargs (str): 3D U-Net parameters
        grid_resolution (int): defined resolution for grid feature 
        n_blocks (int): number of blocks ResNetBlockFC layers
        map2local (function): map global coordintes to local ones
    '''

    def __init__(self, c_dim=32, dim=3, hidden_dim=32, 
                 unet3d_kwargs=None, 
                 grid_resolution=32, n_blocks=5,
                 map2local=None):
        super().__init__()
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.unet3d = unet3d.UNet3D(**unet3d_kwargs)
        self.reso_grid = grid_resolution
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.map2local = map2local


    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone())
        index = coordinate3d_index(p_nor, self.reso_grid)
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)
        fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, index, c):
        bs, fea_dim = c.size(0), c.size(2)

        fea = scatter_max(c.permute(0, 2, 1), index, dim_size=self.reso_grid**3)
        fea = fea[0] # for scatter_max
        # gather feature back to points
        fea = fea.gather(dim=2, index=index.expand(-1, fea_dim, -1))

        return fea.permute(0, 2, 1)


    def forward(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = normalize_3d_coordinate(p.clone())
        index = coordinate3d_index(coord, self.reso_grid)

        # xxxx8888
        if self.map2local:
            pp = self.map2local(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = self.generate_grid_features(p, c)

        return fea

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling feature strategy, bilinear|nearest
    '''

    def __init__(self, dim=3, c_dim=128, out_dim=3,
                 hidden_size=256, n_blocks=5, sample_mode='bilinear', map2local=None):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        self.fc_p = nn.Linear(dim, hidden_size)
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.actvn = F.relu

        self.sample_mode = sample_mode
        self.map2local = map2local
        self.out_dim = out_dim
    

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone())
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', 
                                    align_corners=True, 
                                    mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c):
        batch_size = p.shape[0]
        c = self.sample_grid_feature(p, c)
        c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)
        
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
    
        # xxxx8888
        if self.out_dim > 3:
            out = out.reshape(batch_size, -1, 3)
            
        return out

class Encode2Points(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 3
        c_dim = 32
        out_dim = 3
        self.num_offset = 7

        encoder_kwargs = {
        	'hidden_dim': 32,
            'grid_resolution': 32,
            'unet3d_kwargs': {'num_levels': 3, 'f_maps': 32, 'in_channels': 32, 'out_channels': 32}
        }

        decoder_kwargs = {
        	'sample_mode': 'bilinear',
        	'hidden_size': 32
        }

        # local mapping
        unit_size = 1 / encoder_kwargs['grid_resolution'] # 32
        local_mapping = map2local(unit_size)

        self.encoder = LocalPoolPointnet(
            dim=dim, c_dim=c_dim, map2local=local_mapping,
            **encoder_kwargs
        )

        # decoder for normal prediction
        self.decoder_normal = LocalDecoder(
            dim=dim, c_dim=c_dim, out_dim=out_dim,
            **decoder_kwargs)

        # decoder for offset prediction
        self.decoder_offset = LocalDecoder(
    	    dim=dim, c_dim=c_dim, out_dim=self.num_offset * out_dim,
            map2local=local_mapping, **decoder_kwargs)

        self.s_off = 0.001

        # self = Encode2Points(
        #   (encoder): LocalPoolPointnet(
        #     (fc_pos): Linear(in_features=3, out_features=64, bias=True)
        #     (blocks): ModuleList(
        #       (0-4): 5 x ResnetBlockFC(
        #         (fc_0): Linear(in_features=64, out_features=32, bias=True)
        #         (fc_1): Linear(in_features=32, out_features=32, bias=True)
        #         (actvn): ReLU()
        #         (shortcut): Linear(in_features=64, out_features=32, bias=False)
        #       )
        #     )
        #     (fc_c): Linear(in_features=32, out_features=32, bias=True)
        #     (actvn): ReLU()
        #     (unet3d): UNet3D(
        #       (encoders): ModuleList(
        #         (0): Encoder(
        #           (basic_module): DoubleConv(
        #             (SingleConv1): SingleConv(
        #               (groupnorm): GroupNorm(8, 32, eps=1e-05, affine=True)
        #               (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #             (SingleConv2): SingleConv(
        #               (groupnorm): GroupNorm(8, 32, eps=1e-05, affine=True)
        #               (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #           )
        #         )
        #         (1): Encoder(
        #           (pooling): MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False)
        #           (basic_module): DoubleConv(
        #             (SingleConv1): SingleConv(
        #               (groupnorm): GroupNorm(8, 32, eps=1e-05, affine=True)
        #               (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #             (SingleConv2): SingleConv(
        #               (groupnorm): GroupNorm(8, 32, eps=1e-05, affine=True)
        #               (conv): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #           )
        #         )
        #         (2): Encoder(
        #           (pooling): MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False)
        #           (basic_module): DoubleConv(
        #             (SingleConv1): SingleConv(
        #               (groupnorm): GroupNorm(8, 64, eps=1e-05, affine=True)
        #               (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #             (SingleConv2): SingleConv(
        #               (groupnorm): GroupNorm(8, 64, eps=1e-05, affine=True)
        #               (conv): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #           )
        #         )
        #       )
        #       (decoders): ModuleList(
        #         (0): Decoder(
        #           (upsampling): Upsampling()
        #           (basic_module): DoubleConv(
        #             (SingleConv1): SingleConv(
        #               (groupnorm): GroupNorm(8, 192, eps=1e-05, affine=True)
        #               (conv): Conv3d(192, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #             (SingleConv2): SingleConv(
        #               (groupnorm): GroupNorm(8, 64, eps=1e-05, affine=True)
        #               (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #           )
        #         )
        #         (1): Decoder(
        #           (upsampling): Upsampling()
        #           (basic_module): DoubleConv(
        #             (SingleConv1): SingleConv(
        #               (groupnorm): GroupNorm(8, 96, eps=1e-05, affine=True)
        #               (conv): Conv3d(96, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #             (SingleConv2): SingleConv(
        #               (groupnorm): GroupNorm(8, 32, eps=1e-05, affine=True)
        #               (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #               (ReLU): ReLU(inplace=True)
        #             )
        #           )
        #         )
        #       )
        #       (final_conv): Conv3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        #     )
        #   )
        #   (decoder_normal): LocalDecoder(
        #     (fc_c): ModuleList(
        #       (0-4): 5 x Linear(in_features=32, out_features=32, bias=True)
        #     )
        #     (fc_p): Linear(in_features=3, out_features=32, bias=True)
        #     (blocks): ModuleList(
        #       (0-4): 5 x ResnetBlockFC(
        #         (fc_0): Linear(in_features=32, out_features=32, bias=True)
        #         (fc_1): Linear(in_features=32, out_features=32, bias=True)
        #         (actvn): ReLU()
        #       )
        #     )
        #     (fc_out): Linear(in_features=32, out_features=3, bias=True)
        #   )
        #   (decoder_offset): LocalDecoder(
        #     (fc_c): ModuleList(
        #       (0-4): 5 x Linear(in_features=32, out_features=32, bias=True)
        #     )
        #     (fc_p): Linear(in_features=3, out_features=32, bias=True)
        #     (blocks): ModuleList(
        #       (0-4): 5 x ResnetBlockFC(
        #         (fc_0): Linear(in_features=32, out_features=32, bias=True)
        #         (fc_1): Linear(in_features=32, out_features=32, bias=True)
        #         (actvn): ReLU()
        #       )
        #     )
        #     (fc_out): Linear(in_features=32, out_features=21, bias=True)
        #   )
        # )


    def forward(self, p):
        # p.size() -- [1, 3000, 3]
       
        batch_size = p.size(0)
        points = p.clone()

        # encode the input point cloud to a feature volume
        c = self.encoder(p) # c.size() -- [1, 32, 32, 32, 32]
        offset = self.decoder_offset(p, c) # offset.size() -- [1, 21000, 3]
        # points.size() -- [1, 3000, 3]

        # more than one offset is predicted per-point
        points = points.repeat(1, 1, self.num_offset).reshape(batch_size, -1, 3)
        points = points + self.s_off * offset # self.s_off -- 0.001
        points = torch.clamp(points, 0.0, 0.99)

        normals = self.decoder_normal(points, c)

        # (Pdb) pp points.size() -- [1, 21000, 3]
        # (Pdb) pp normals.size() -- [1, 21000, 3]
        
        return points, normals
    

class MeshSDF(nn.Module):
    def __init__(self, res=(128, 128, 128), sigma=2):
        super(MeshSDF, self).__init__()
        self.enc = Encode2Points()
        self.dpsr = dpsr.DPSR(res, sigma=sigma)
        self.load_weights()

    def load_weights(self, model_path="models/points_mesh.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.enc.load_state_dict(torch.load(checkpoint))


    def forward(self, points):
    	points, normals = self.enc(points)
    	chi = self.dpsr(points, normals)

    	return chi, points, normals
