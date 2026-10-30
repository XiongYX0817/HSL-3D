import os
import sys
import importlib
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import o_voxel
from easydict import EasyDict as edict
from functools import partial

from octree_utils.voxelize import process_pointcloud, rebuild_octree
from octree_utils.normals import face_areas_normals, vertex_normals, sample_points_and_normals
from octree_utils.utils import *

import torch

def read_dual_grid(filepath, use_sparse_tensor=True):
    coords, attr = o_voxel.io.read_vxz(filepath, num_threads=4)
    return coords

def _octreelize(file, metadatum, mesh_dump_root, root, resolution):
    sha256 = metadatum['sha256']
    try:
        # check if already processed
        if os.path.exists(os.path.join(root, f'octrees_{resolution}', f'{sha256}.pt')):
            try:
                info = torch.load(os.path.join(root, f'octrees_{resolution}', f'{sha256}.pt'), map_location="cpu")
                return {'sha256': sha256, 'octreelized': True}
            except Exception as e:
                print(f'Error reading {sha256}.pt: {e}')

        # process mesh
        with open(os.path.join(mesh_dump_root, 'mesh_dumps', f'{sha256}.pickle'), 'rb') as f:
            dump = pickle.load(f)
        start = 0
        vertices = []
        faces = []
        for obj in dump['objects']:
            if obj['vertices'].size == 0 or obj['faces'].size == 0:
                continue
            vertices.append(obj['vertices'])
            faces.append(obj['faces'] + start)
            start += len(obj['vertices'])
        vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float().cuda()
        faces = torch.from_numpy(np.concatenate(faces, axis=0)).long().cuda()
        vertices_min = vertices.min(dim=0)[0]
        vertices_max = vertices.max(dim=0)[0]
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
        
        areas, face_normals = face_areas_normals(vertices, faces)
        vnormals = vertex_normals(vertices, faces, face_normals, areas)
        points, normals = sample_points_and_normals(vertices, faces, vnormals, areas, n_samples=2_000_000)
        assert torch.isfinite(points).all()

        gt_finest_voxel = None
        if os.path.exists(os.path.join(root, 'dual_grid_256', f'{sha256}.vxz')):
            gt_finest_voxel = read_dual_grid(os.path.join(root, 'dual_grid_256', f'{sha256}.vxz'))
            gt_finest_voxel = torch.unique(gt_finest_voxel // (64 // resolution), dim=0)
        octree_info = process_pointcloud(points, normals, level=3, res=resolution, threshold=0.5, gt_finest_voxel=gt_finest_voxel, device='cuda')

        torch.save(octree_info, os.path.join(root, f'octrees_{resolution}', f'{sha256}.pt'))
        # export_octree_to_obj(rebuild_octree(octree_info, N=resolution, level=3), N=resolution, filename=os.path.join(root, f'octrees_{resolution}', f'{sha256}_octree.obj'))

        return {'sha256': sha256, 'octreelized': True}
    except Exception as e:
        print(f'Error processing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to load mesh dumps')
    parser.add_argument('--octreelize_root', type=str, default=None,
                        help='Directory to save dual grids')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.octreelize_root = opt.octreelize_root or opt.root

    os.makedirs(os.path.join(opt.octreelize_root, f'octrees_{opt.resolution}', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['mesh_dumped'] == True]
        mask = np.zeros(len(metadata), dtype=bool)
        if 'octreelized' in metadata.columns:
            metadata = metadata[metadata['octreelized'] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # # filter out objects that are already processed
    # sha256_list = os.listdir(os.path.join(opt.octreelize_root, f'octrees_{opt.resolution}'))
    # sha256_list = [os.path.splitext(f)[0] for f in sha256_list if f.endswith('.pt')]
    # for sha256 in sha256_list:
    #     records.append({'sha256': sha256, 'octreelized': True})
    # print(f'Found {len(sha256_list)} octreelized mesh')
    # metadata = metadata[~metadata['sha256'].isin(sha256_list)]
    
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_octreelize, root=opt.octreelize_root, mesh_dump_root=opt.mesh_dump_root, resolution=opt.resolution)
    octreelized = dataset_utils.foreach_instance(metadata, None, func, max_workers=opt.max_workers, no_file=True, desc='Octreelize')
    # octreelized = pd.concat([octreelized, pd.DataFrame.from_records(records)])
    octreelized.to_csv(os.path.join(opt.octreelize_root, f'octrees_{opt.resolution}', 'new_records', f'part_{opt.rank}.csv'), index=False)
    