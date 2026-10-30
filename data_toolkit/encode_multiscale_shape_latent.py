import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

import o_voxel
import hsl3d.models as models
import hsl3d.modules.sparse as sp
from hsl3d.utils.octree_utils import rebuild_octree
from hsl3d.utils.multiscale_shape_latent_utils import (
    LATENT_LEVELS,
    PACK_FORMAT,
    PACK_LEVEL_FORMAT,
)

torch.set_grad_enabled(False)

def is_valid_sparse_tensor(tensor: sp.SparseTensor) -> bool:
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def _normalize_pack_metadata(pack: dict) -> dict:
    normalized = {}
    for key, value in pack.items():
        if isinstance(value, np.ndarray) and value.ndim == 0:
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def _validate_level_format(pack: Mapping[str, object]) -> None:
    if str(pack.get('format')) != PACK_FORMAT:
        raise ValueError(f"Unsupported latent format: {pack.get('format')}")
    if str(pack.get('level_format')) != PACK_LEVEL_FORMAT:
        raise ValueError(f"Unsupported level format: {pack.get('level_format')}")


def get_level_counts_from_pack(pack: Mapping[str, object]) -> Dict[int, int]:
    _validate_level_format(pack)
    levels = np.asarray(pack['level'])
    valid_levels = np.asarray(LATENT_LEVELS, dtype=levels.dtype if levels.size > 0 else np.int64)
    unknown_levels = np.setdiff1d(np.unique(levels), valid_levels)
    if unknown_levels.size > 0:
        raise ValueError(f'Unexpected latent levels: {unknown_levels.tolist()}')
    counts = {level: int((levels == level).sum()) for level in LATENT_LEVELS}
    return counts


def clear_cuda_error() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def read_dual_grid(dual_grid_root: str, resolution: int, sha256: str) -> Tuple[sp.SparseTensor, sp.SparseTensor]:
    coords, attr = o_voxel.io.read_vxz(
        os.path.join(dual_grid_root, f'dual_grid_{resolution}', f'{sha256}.vxz'),
        num_threads=4,
    )
    vertices = sp.SparseTensor(
        (attr['vertices'] / 255.0).float(),
        torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
    )
    intersected = vertices.replace(torch.cat([
        attr['intersected'] % 2,
        attr['intersected'] // 2 % 2,
        attr['intersected'] // 4 % 2,
    ], dim=-1).bool())
    return vertices, intersected


def resolve_octree_path(octree_root: str, resolution: int, sha256: str) -> str:
    direct_path = os.path.join(octree_root, f'{sha256}.pt')
    if os.path.exists(direct_path):
        return direct_path

    if resolution % 64 != 0:
        raise ValueError(
            f'Cannot infer octree directory for resolution={resolution}. '
            'Expected a multiple of 64 such as 256, 512, or 1024.'
        )

    octree_resolution = resolution // 64
    candidate = os.path.join(octree_root, f'octrees_{octree_resolution}', f'{sha256}.pt')
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f'Octree file not found for {sha256}. Tried: {direct_path}, {candidate}'
    )


def read_octree_frontier(octree_root: str, resolution: int, sha256: str) -> Dict[int, torch.Tensor]:
    octree_info = torch.load(resolve_octree_path(octree_root, resolution, sha256), map_location='cpu')
    octree_dict = rebuild_octree(octree_info)
    frontier = {}
    for level in LATENT_LEVELS:
        coords_xyz = torch.as_tensor(octree_dict[level], dtype=torch.int32)
        if coords_xyz.numel() == 0:
            coords_xyz = coords_xyz.reshape(0, 3)
        frontier[level] = torch.cat([
            torch.zeros((coords_xyz.shape[0], 1), dtype=torch.int32),
            coords_xyz,
        ], dim=1).contiguous()
    return frontier


def build_latent_name(opt) -> str:
    if opt.latent_name is not None:
        return opt.latent_name
    return f'{opt.model_dir}_{opt.ckpt}_{opt.resolution}'


def load_model_bundle(opt):
    cfg_path = os.path.join(opt.model_root, opt.model_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'Config not found: {cfg_path}')
    cfg = edict(json.load(open(cfg_path, 'r')))

    encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
    encoder_ckpt_path = os.path.join(opt.model_root, opt.model_dir, 'ckpts', f'encoder_{opt.ckpt}.pt')
    encoder.load_state_dict(
        torch.load(encoder_ckpt_path, map_location='cpu', weights_only=True)
    )
    encoder.eval()

    return encoder

def pack_multiscale_latents(latents: Mapping[int, sp.SparseTensor]) -> Tuple[dict, Dict[int, int]]:
    feats_list = []
    coords_list = []
    level_list = []
    level_counts = {}
    for latent_level in LATENT_LEVELS:
        latent = latents[latent_level]
        coords = latent.coords[:, 1:].detach().cpu().numpy().astype(np.uint16, copy=False)
        feats = latent.feats.detach().cpu().numpy().astype(np.float32, copy=False)
        level_arr = np.full((coords.shape[0],), latent_level, dtype=np.uint8)

        if coords.shape[0] > 0:
            order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
            coords = coords[order]
            feats = feats[order]
            level_arr = level_arr[order]

        feats_list.append(feats)
        coords_list.append(coords)
        level_list.append(level_arr)
        level_counts[latent_level] = int(coords.shape[0])

    feat_dim = int(next(iter(latents.values())).feats.shape[1]) if len(latents) > 0 else 0
    feats = np.concatenate(feats_list, axis=0) if len(feats_list) > 0 else np.empty((0, feat_dim), dtype=np.float32)
    coords = np.concatenate(coords_list, axis=0) if len(coords_list) > 0 else np.empty((0, 3), dtype=np.uint16)
    levels = np.concatenate(level_list, axis=0) if len(level_list) > 0 else np.empty((0,), dtype=np.uint8)

    pack = {
        'format': PACK_FORMAT,
        'level_format': PACK_LEVEL_FORMAT,
        'feats': feats,
        'coords': coords,
        'level': levels,
    }
    return pack, level_counts


def save_pack(save_path: str, pack: dict, storage_format: str) -> None:
    if storage_format == 'npy':
        np.save(save_path, pack, allow_pickle=True)
    elif storage_format == 'npz':
        np.savez_compressed(save_path, **pack)
    else:
        raise ValueError(f'Unknown storage format: {storage_format}')


def load_existing_pack(path: str, storage_format: str) -> dict:
    if storage_format == 'npy':
        return _normalize_pack_metadata(np.load(path, allow_pickle=True).item())
    if storage_format == 'npz':
        data = np.load(path, allow_pickle=False)
        return _normalize_pack_metadata({k: data[k] for k in data.files})
    raise ValueError(f'Unknown storage format: {storage_format}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Directory to save the metadata')
    parser.add_argument('--dual_grid_root', type=str, default=None, help='Directory containing dual grids')
    parser.add_argument('--octree_root', type=str, default=None, help='Directory containing octree files or octrees_<resolution//64> subdirectories')
    parser.add_argument('--shape_latent_root', type=str, default=None, help='Directory to save multiscale shape latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None, help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, required=True, help='Sparse voxel resolution / dual grid resolution')
    parser.add_argument('--model_root', type=str, required=True, help='Root directory of trained models')
    parser.add_argument('--model_dir', type=str, required=True, help='Training output directory name under model_root')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint tag to load, e.g. step0010000')
    parser.add_argument('--latent_name', type=str, default=None, help='Optional output latent directory name')
    parser.add_argument('--instances', type=str, default=None, help='Instances to process')
    parser.add_argument('--storage_format', type=str, default='npz', choices=['npy', 'npz'], help='Per-sample storage format')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.octree_root = opt.octree_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root

    encoder = load_model_bundle(opt)
    data_resolution = int(opt.resolution)
    latent_name = build_latent_name(opt)
    latent_dir = os.path.join(opt.shape_latent_root, 'shape_latents', latent_name)
    os.makedirs(os.path.join(latent_dir, 'new_records'), exist_ok=True)

    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')

    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.dual_grid_root, f'dual_grid_{data_resolution}', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.dual_grid_root, f'dual_grid_{data_resolution}', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(latent_dir, 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(latent_dir, 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['dual_grid_converted'] == True]
        if 'shape_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['shape_latent_encoded'] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r', encoding='utf-8') as fp:
                instances = fp.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    extension = '.npy' if opt.storage_format == 'npy' else '.npz'

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(metadata), desc='Filtering existing objects') as pbar:
        def check_sha256(sha256: str) -> None:
            save_path = os.path.join(latent_dir, f'{sha256}{extension}')
            if os.path.exists(save_path):
                pack = load_existing_pack(save_path, opt.storage_format)
                level_counts = get_level_counts_from_pack(pack)
                records.append({
                    'sha256': sha256,
                    'shape_latent_encoded': True,
                    'shape_latent_tokens': int(pack['coords'].shape[0]),
                    'shape_latent_level0_tokens': int(level_counts.get(0, 0)),
                    'shape_latent_level1_tokens': int(level_counts.get(1, 0)),
                    'shape_latent_level2_tokens': int(level_counts.get(2, 0)),
                    'shape_latent_storage': opt.storage_format,
                })
            pbar.update()

        executor.map(check_sha256, metadata['sha256'].values)
        executor.shutdown(wait=True)

    existing_sha256 = set(r['sha256'] for r in records)
    print(f'Found {len(existing_sha256)} processed objects')
    metadata = metadata[~metadata['sha256'].isin(existing_sha256)]

    print(f'Processing {len(metadata)} objects...')
    sha256s = list(metadata['sha256'].values)
    load_queue = Queue(maxsize=32)

    with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
        def loader(sha256: str) -> None:
            try:
                vertices, intersected = read_dual_grid(opt.dual_grid_root, data_resolution, sha256)
                octree_frontier = read_octree_frontier(
                    opt.octree_root,
                    data_resolution,
                    sha256,
                )
                load_queue.put((sha256, vertices, intersected, octree_frontier))
            except Exception as e:
                print(f'[Loader Error] {sha256}: {e}')
                load_queue.put((sha256, None, None, None))

        loader_executor.map(loader, sha256s)

        def saver(sha256: str, pack: dict, level_counts: Dict[int, int]) -> None:
            save_path = os.path.join(latent_dir, f'{sha256}{extension}')
            save_pack(save_path, pack, opt.storage_format)
            records.append({
                'sha256': sha256,
                'shape_latent_encoded': True,
                'shape_latent_tokens': int(pack['coords'].shape[0]),
                'shape_latent_level0_tokens': int(level_counts.get(0, 0)),
                'shape_latent_level1_tokens': int(level_counts.get(1, 0)),
                'shape_latent_level2_tokens': int(level_counts.get(2, 0)),
                'shape_latent_storage': opt.storage_format,
            })

        for _ in tqdm(range(len(sha256s)), desc='Extracting multiscale latents'):
            sha256 = None
            num_voxels = 0
            try:
                sha256, vertices, intersected, octree_frontier = load_queue.get()
                if vertices is None or intersected is None:
                    print(f'[Skip] {sha256}: Failed to load input')
                    continue

                num_voxels = int(vertices.feats.shape[0])
                if not (is_valid_sparse_tensor(vertices) and is_valid_sparse_tensor(intersected)):
                    print(f'[Skip] {sha256}: NaN/Inf in input')
                    continue

                vertices = vertices.cuda()
                intersected = intersected.cuda()
                structure = {
                    latent_level: coords.cuda().contiguous()
                    for latent_level, coords in octree_frontier.items()
                }
                latents = encoder(
                    vertices,
                    intersected,
                    structure=structure,
                )
                torch.cuda.synchronize()

                if not isinstance(latents, Mapping):
                    raise TypeError('Stage II encoder must return a level-to-latent mapping')
                latents = {int(level): latent for level, latent in latents.items()}
                if set(latents) != set(LATENT_LEVELS):
                    raise ValueError(
                        f'Stage II encoder must return levels {list(LATENT_LEVELS)}, got {sorted(latents)}'
                    )
                if any(not torch.isfinite(z.feats).all() for z in latents.values()):
                    print(f'[Skip] {sha256}: Non-finite latent features')
                    clear_cuda_error()
                    continue

                pack, level_counts = pack_multiscale_latents(latents)
                saver_executor.submit(saver, sha256, pack, level_counts)
            except Exception as e:
                print(f'[Error] {sha256} ({num_voxels} voxels): {e}')
                clear_cuda_error()
                continue

        saver_executor.shutdown(wait=True)

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(latent_dir, 'new_records', f'part_{opt.rank}.csv'), index=False)
