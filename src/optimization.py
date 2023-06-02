import time, os
import numpy as np
import torch
import trimesh

from src.dpsr import DPSR
from src.model import PSR2Mesh
from src.utils import verts_on_largest_mesh, export_pointcloud, mc_from_psr
from pytorch3d.loss import chamfer_distance
# import open3d as o3d
import todos

import pdb

class Trainer(object):
    '''
    Args:
        cfg       : config file
        optimizer : pytorch optimizer object
        device    : pytorch device
    '''

    def __init__(self, cfg, optimizer, device=None):
        self.optimizer = optimizer # Adam
        self.device = device
        self.cfg = cfg
        self.psr2mesh = PSR2Mesh.apply

        # initialize DPSR
        self.dpsr = DPSR(res=(cfg['model']['grid_res'], 
                            cfg['model']['grid_res'], 
                            cfg['model']['grid_res']), 
                        sig=cfg['model']['psr_sigma'])
        self.dpsr = self.dpsr.to(device)


    def train_step(self, data, inputs, it):
        ''' Performs a training step.

        Args:
            data (dict)              : data dictionary
            inputs (torch.tensor)    : input point clouds
            it (int)                 : the number of iterations
        '''
        self.optimizer.zero_grad()
        loss, loss_each = self.compute_loss(inputs, data, it)

        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_each

    def compute_loss(self, inputs, data, it=0):
        '''  Compute the loss.
        Args:
            data (dict)              : data dictionary
            inputs (torch.tensor)    : input point clouds
            it (int)                 : the number of iterations
        '''

        res = self.cfg['model']['grid_res']
        
        # source oriented point clouds to PSR grid
        psr_grid, points, normals = self.pcl2psr(inputs)
        
        # build mesh
        v, f, n = self.psr2mesh(psr_grid)
        
        # the output is in the range of [0, 1), we make it to the real range [0, 1]. 
        # This is a hack for our DPSR solver
        v = v * res / (res-1) 

        points = points * 2. - 1.
        v = v * 2. - 1. # within the range of (-1, 1)

        loss = 0
        loss_each = {}

        # compute loss
        # self.cfg['train']['w_chamfer'] -- 1
        loss_ = 1.0 * self.compute_3d_loss(v, data)
        loss_each['chamfer'] = loss_
        loss += loss_
    
        return loss, loss_each


    def pcl2psr(self, inputs):
        '''  Convert an oriented point cloud to PSR indicator grid
        Args:
            inputs (torch.tensor): input oriented point clouds
        '''

        points, normals = inputs[...,:3], inputs[...,3:]
        points = torch.sigmoid(points)

        # DPSR to get grid
        psr_grid = self.dpsr(points, normals).unsqueeze(1)
        psr_grid = torch.tanh(psr_grid)

        return psr_grid, points, normals

    def compute_3d_loss(self, v, data):
        '''  Compute the loss for point clouds.
        Args:
            v (torch.tensor)         : mesh vertices
            data (dict)              : data dictionary
        '''

        pts_gt = data.get('target_points')
        loss, _ = chamfer_distance(v, pts_gt)

        return loss
    
    def point_resampling(self, inputs):
        # xxxx8888
        '''  Resample points
        Args:
            inputs (torch.tensor): oriented point clouds
        '''
        # inputs.size() -- [1, 20000, 6]
    
        psr_grid, points, normals = self.pcl2psr(inputs)
        
        # shortcuts
        n_grow = 2000 # self.cfg['train']['n_grow_points']

        # [hack] for points resampled from the mesh from marching cubes, 
        # we need to divide by s instead of (s-1), and the scale is correct.
        # verts, faces, _ = mc_from_psr(psr_grid, real_scale=False, zero_level=0)
        verts, faces, _ = mc_from_psr(psr_grid, zero_level=0)

        # find the largest component
        # xxxx8888 ?
        pts_mesh, faces_mesh = verts_on_largest_mesh(verts, faces)

        # sample vertices only from the largest component, not from fragments
        mesh = trimesh.Trimesh(vertices=pts_mesh, faces=faces_mesh)
        pi, face_idx = mesh.sample(n_grow + points.shape[1], return_index=True)
        normals_i = mesh.face_normals[face_idx].astype('float32')
        pts_mesh = torch.tensor(pi.astype('float32')).to(self.device)[None]
        n_mesh = torch.tensor(normals_i).to(self.device)[None]

        points, normals = pts_mesh, n_mesh
        print('{} total points are resampled'.format(points.shape[1]))
    
        # update inputs
        points = torch.log(points / (1 - points)) # inverse sigmoid
        inputs = torch.cat([points, normals], dim=-1)
        inputs.requires_grad = True

        # inputs.size() -- [1, 22000, 6]

        return inputs

    def save_mesh_pointclouds(self, inputs, epoch, center=None, scale=None):
        '''  Save meshes and point clouds.
        Args:
            inputs (torch.tensor)       : source point clouds
            epoch (int)                 : the number of iterations
            center (numpy.array)        : center of the shape
            scale (numpy.array)         : scale of the shape
        '''

        # (Pdb) inputs.size() -- [1, 20000, 6]
        # (Pdb) inputs.min() -- -1.0020, inputs.max() -- 1.0018, inputs.mean() -- -0.0031
        # epoch = 0
        # center = array([0.0135, 0.0121, 5.1243], dtype=float32)
        # scale = array([58.9441], dtype=float32)


        psr_grid, points, normals = self.pcl2psr(inputs)
        
        dir_pcl = self.cfg['train']['dir_pcl']
        p = points.squeeze(0).detach().cpu().numpy()
        p = p * 2 - 1
        n = normals.squeeze(0).detach().cpu().numpy()
        if scale is not None:
            p *= scale
        if center is not None:
            p += center
        filename = os.path.join(dir_pcl, '{:04d}.ply'.format(epoch))
        todos.data.save_3dply(torch.from_numpy(p), torch.from_numpy(n), filename)

        # export_pointcloud(os.path.join(dir_pcl, 'o{:04d}.ply'.format(epoch)), p, n)

        dir_mesh = self.cfg['train']['dir_mesh']
        with torch.no_grad():
            v, f, _ = mc_from_psr(psr_grid,
                    zero_level=self.cfg['data']['zero_level'], real_scale=True) # self.cfg['data']['zero_level'] -- 0
            v = v * 2 - 1

            v = v.detach().cpu().numpy()
            f = f.detach().cpu().numpy()
            if scale is not None:
                v *= scale
            if center is not None:
                v += center

        filename = os.path.join(dir_mesh, '{:04d}.obj'.format(epoch))
        todos.data.save_3dobj(torch.from_numpy(v), torch.from_numpy(f.astype(np.float32)).long(), filename)

        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(v)
        # mesh.triangles = o3d.utility.Vector3iVector(f)
        # outdir_mesh = os.path.join(dir_mesh, 'o{:04d}.ply'.format(epoch))
        # o3d.io.write_triangle_mesh(outdir_mesh, mesh)
