import sys, os
import argparse
from src.utils import load_config
import subprocess
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import pdb

def main():

    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')

    args, unknown = parser.parse_known_args() 
    cfg = load_config(args.config, 'configs/default.yaml')

    resolutions=[32, 64, 128, 256]
    iterations=[1000, 1000, 1000, 200]
    lrs=[2e-3, 2e-3*0.7, 2e-3*(0.7**2), 2e-3*(0.7**3)] # reduce lr

    for idx,(res, epoch, lr) in enumerate(zip(resolutions, iterations, lrs)):
        # print(idx, res, epoch, lr)
        # 0 32 1000 0.002
        # 1 64 1000 0.0014
        # 2 128 1000 0.00098
        # 3 256 200 0.0006859999999999999


        if res>cfg['model']['grid_res']:
            continue

        psr_sigma= 2 if res<=128 else 3
        
        if res > 128:
            psr_sigma = 5 if 'thingi_noisy' in args.config else 3

        out_dir = os.path.join(cfg['train']['out_dir'], 'res_%d'%res)
        
        input_mesh='None' if idx==0 else os.path.join(cfg['train']['out_dir'],
                        'res_%d' % (resolutions[idx-1]), 
                        'vis/mesh', '%04d.ply' % (iterations[idx-1]))
        
        # cmd = 'export MKL_SERVICE_FORCE_INTEL=1 && '
        cmd = ""
        cmd += f"python optim.py {args.config}"
        cmd += f" --model:grid_res {res} --model:psr_sigma {psr_sigma}"
        cmd += f" --train:input_mesh {input_mesh} --train:out_dir {out_dir}"
        cmd += f" --train:total_epochs {epoch} --train:lr_pcl {lr}"

        print("Step", idx+1, ":", cmd)
        os.system(cmd)

if __name__=="__main__":
    main()
