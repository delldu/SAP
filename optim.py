import torch
import trimesh
import argparse, time, os

import numpy as np; np.set_printoptions(precision=4)
import open3d as o3d

from src.optimization import Trainer
from src.utils import load_config, update_config, initialize_logger, \
    get_learning_rate_schedules, adjust_learning_rate, AverageMeter, export_pointcloud
from plyfile import PlyData
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

import pdb

def main():
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1457, metavar='S', 
                        help='random seed')
    
    args, unknown = parser.parse_known_args() 

    cfg = load_config(args.config, 'configs/default.yaml')
    cfg = update_config(cfg, unknown)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_type = cfg['data']['data_type']
    data_class = cfg['data']['class']

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    logger = initialize_logger(cfg)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # shutil.copyfile(args.config, 
    #                 os.path.join(cfg['train']['out_dir'], 'config.yaml'))

    data_path = cfg['data']['data_path'] # 'data/demo/wheel.ply'
    plydata = PlyData.read(data_path)
    vertices = np.stack([plydata['vertex']['x'],
                            plydata['vertex']['y'],
                            plydata['vertex']['z']], axis=1)
    normals = np.stack([plydata['vertex']['nx'],
                        plydata['vertex']['ny'],
                        plydata['vertex']['nz']], axis=1)
    center = vertices.mean(0)
    scale = np.max(np.max(np.abs(vertices - center), axis=0))
    vertices -= center
    vertices /= scale
    vertices *= 0.9

    target_pts = torch.tensor(vertices, device=device)[None].float()


    if not torch.is_tensor(center):
        center = torch.from_numpy(center)
    if not torch.is_tensor(scale):
        scale = torch.from_numpy(np.array([scale]))

    data = {'target_points': target_pts, 'gt_mesh': None} # no GT mesh

    # initialize the source point cloud given an input mesh
    # Step 1: cfg['train']['input_mesh'] -- None
    # Step 2: cfg['train']['input_mesh'] -- out/demo_optim/res_32/vis/mesh/1000.ply
    # Step 3: cfg['train']['input_mesh'] -- out/demo_optim/res_64/vis/mesh/1000.ply

    if 'input_mesh' in cfg['train'].keys() and os.path.isfile(cfg['train']['input_mesh']):
        if cfg['train']['input_mesh'].split('/')[-2] == 'mesh':
            mesh_tmp = trimesh.load_mesh(cfg['train']['input_mesh'])
            verts = torch.from_numpy(mesh_tmp.vertices[None]).float().to(device)
            faces = torch.from_numpy(mesh_tmp.faces[None]).to(device)
            mesh = Meshes(verts=verts, faces=faces)
            points, normals = sample_points_from_meshes(mesh, 
                        num_samples=cfg['data']['num_points'], return_normals=True)
            # mesh is saved in the original scale of the gt
            points -= center.float().to(device)
            points /= scale.float().to(device)
            points *= 0.9
            # make sure the points are within the range of [0, 1)
            points = points / 2. + 0.5
        else:
            # directly initialize from a point cloud
            pcd = o3d.io.read_point_cloud(cfg['train']['input_mesh'])
            points = torch.from_numpy(np.array(pcd.points)[None]).float().to(device)
            normals = torch.from_numpy(np.array(pcd.normals)[None]).float().to(device)
            points -= center.float().to(device)
            points /= scale.float().to(device)
            points *= 0.9
            points = points / 2. + 0.5
    else: #! initialize our source point cloud from a sphere
        sphere_radius = cfg['model']['sphere_radius'] # 0.2
        sphere_mesh = trimesh.creation.uv_sphere(radius=sphere_radius, count=[256,256])
        # sphere_mesh -- <trimesh.Trimesh(vertices.shape=(130816, 3), faces.shape=(259588, 3))>

        # cfg['data']['num_points'] -- 20000
        points, idx = sphere_mesh.sample(cfg['data']['num_points'], return_index=True)

        points += 0.5 # make sure the points are within the range of [0, 1)
        normals = sphere_mesh.face_normals[idx]
        points = torch.from_numpy(points).unsqueeze(0).to(device) # [1, 20000, 3]
        normals = torch.from_numpy(normals).unsqueeze(0).to(device) # [1, 20000, 3]

    
    points = torch.log(points/(1-points)) # inverse sigmoid
    inputs = torch.cat([points, normals], axis=-1).float()
    inputs.requires_grad = True

    # initialize optimizer
    cfg['train']['schedule']['pcl']['initial'] = cfg['train']['lr_pcl']
    print('Initial learning rate:', cfg['train']['schedule']['pcl']['initial'])

    if 'schedule' in cfg['train']:
        # cfg['train']['schedule']
        # {'pcl': {'initial': '0.002', 'interval': 700, 'factor': 0.5, 'final': '1e-3'}}
        lr_schedules = get_learning_rate_schedules(cfg['train']['schedule'])
    else:
        lr_schedules = None

    optimizer = torch.optim.Adam([inputs], lr=lr_schedules[0].get_learning_rate(0))
    try:
        # load model
        state_dict = torch.load(os.path.join(cfg['train']['out_dir'], 'model.pt'))
        if ('pcl' in state_dict.keys()) & (state_dict['pcl'] is not None):
            inputs = state_dict['pcl'].to(device)
            inputs.requires_grad = True
        
        optimizer = torch.optim.Adam([inputs], lr=lr_schedules[0].get_learning_rate(state_dict.get('epoch')))
            
        out = "Load model from epoch %d" % state_dict.get('epoch', 0)
        print(out)
        logger.info(out)
    except:
        state_dict = dict()

    start_epoch = state_dict.get('epoch', -1)

    # (Pdb) optimizer
    # Adam (
    # Parameter Group 0
    #     amsgrad: False
    #     betas: (0.9, 0.999)
    #     capturable: False
    #     differentiable: False
    #     eps: 1e-08
    #     foreach: None
    #     fused: None
    #     lr: 0.001
    #     maximize: False
    #     weight_decay: 0
    # )

    trainer = Trainer(cfg, optimizer, device=device)
    runtime = AverageMeter()
    
    # training loop
    for epoch in range(start_epoch+1, cfg['train']['total_epochs']+1):
        # schedule the learning rate
        if (epoch>0) & (lr_schedules is not None):
            if (epoch % lr_schedules[0].interval == 0):
                adjust_learning_rate(lr_schedules, optimizer, epoch)
                # if len(lr_schedules) >1:
                #     print('[epoch {}] net_lr: {}, pcl_lr: {}'.format(epoch, 
                #                         lr_schedules[0].get_learning_rate(epoch), 
                #                         lr_schedules[1].get_learning_rate(epoch)))
                # else:
                #     print('[epoch {}] adjust pcl_lr to: {}'.format(epoch, 
                #                         lr_schedules[0].get_learning_rate(epoch)))
                print('[epoch {}] adjust pcl_lr to: {}'.format(epoch, lr_schedules[0].get_learning_rate(epoch)))


        start = time.time()
        loss, loss_each = trainer.train_step(data, inputs, epoch)
        runtime.update(time.time() - start)

        # cfg['train']['print_every'] -- 10
        if epoch % cfg['train']['print_every'] == 0:
            log_text = ('[Epoch %02d] loss=%.5f') %(epoch, loss)
            for k, l in loss_each.items():
                if l.item() != 0.:
                    log_text += (' loss_%s=%.5f') % (k, l.item())
            
            log_text += (' time=%.3f / %.3f') % (runtime.val, runtime.sum)
            logger.info(log_text)
            print(log_text)

    
        if epoch == cfg['train']['total_epochs']:
            # save outputs
            trainer.save_mesh_pointclouds(inputs, epoch, center.cpu().numpy(), scale.cpu().numpy()*(1/0.9))

            # save checkpoints
            state = {'epoch': epoch}
            if isinstance(inputs, torch.Tensor):
                state['pcl'] = inputs.detach().cpu()
            logger.info("Save model at epoch %d" % epoch)
            torch.save(state, os.path.join(cfg['train']['out_dir'], 'model.pt'))
        
        # resample and gradually add new points to the source pcl
        # cfg['train']['resample_every'] -- 200
        if (epoch > 0) & \
           (epoch % cfg['train']['resample_every'] == 0) & \
           (epoch < cfg['train']['total_epochs']):
                inputs = trainer.point_resampling(inputs)
                optimizer = torch.optim.Adam([inputs], lr=lr_schedules[0].get_learning_rate(epoch))

                trainer = Trainer(cfg, optimizer, device=device)
    

if __name__ == '__main__':
    main()
