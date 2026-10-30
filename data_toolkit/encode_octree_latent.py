import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

import hsl3d.models as models
from hsl3d.utils.octree_utils import octree2tensor


torch.set_grad_enabled(False)


def load_model_bundle(opt):
    config_path = os.path.join(opt.model_root, opt.model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as fp:
        cfg = edict(json.load(fp))
    if 'encoder' not in cfg.models:
        raise ValueError(f'No encoder is defined in {config_path}')

    encoder_cfg = cfg.models.encoder
    encoder = getattr(models, encoder_cfg.name)(**encoder_cfg.args)
    checkpoint_path = os.path.join(
        opt.model_root,
        opt.model_dir,
        'ckpts',
        f'encoder_{opt.ckpt}.pt',
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Encoder checkpoint not found: {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder = encoder.cuda().eval()

    dataset_args = getattr(cfg.dataset, 'args', edict())
    config_resolution = int(getattr(dataset_args, 'resolution', 16))
    config_num_levels = int(getattr(dataset_args, 'level', encoder_cfg.args.in_channels))
    resolution = config_resolution if opt.resolution is None else int(opt.resolution)
    num_levels = config_num_levels if opt.num_levels is None else int(opt.num_levels)

    if int(encoder_cfg.args.in_channels) != num_levels:
        raise ValueError(
            'The number of octree levels must match the encoder input channels: '
            f'{num_levels} != {encoder_cfg.args.in_channels}'
        )
    return cfg, encoder, resolution, num_levels


def build_latent_name(opt, resolution: int) -> str:
    if opt.latent_name is not None:
        return opt.latent_name
    model_name = os.path.basename(os.path.normpath(opt.model_dir))
    return f'{model_name}_{opt.ckpt}_{resolution}'


def load_instance_list(instances: str) -> List[str]:
    if os.path.exists(instances):
        with open(instances, 'r', encoding='utf-8') as fp:
            return [line.strip() for line in fp if line.strip()]
    return [item.strip() for item in instances.split(',') if item.strip()]


def combine_metadata(metadata: pd.DataFrame, path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return metadata
    other = pd.read_csv(path).set_index('sha256')
    return metadata.combine_first(other)


def octree_info_to_tensor(
    octree_info: Dict[int, torch.Tensor],
    base_resolution: int,
    num_levels: int,
) -> torch.Tensor:
    dense_levels = {}
    for level in range(num_levels):
        coords = octree_info.get(level, octree_info.get(str(level)))
        if coords is None:
            raise ValueError(f'Missing octree level {level}')
        coords = torch.as_tensor(coords, dtype=torch.long).reshape(-1, 3)
        level_resolution = base_resolution * 2**level
        if coords.numel() > 0:
            if torch.any(coords < 0) or torch.any(coords >= level_resolution):
                raise ValueError(
                    f'Octree level {level} contains coordinates outside '
                    f'[0, {level_resolution})'
                )
        dense = torch.zeros(
            (1, level_resolution, level_resolution, level_resolution),
            dtype=torch.uint8,
        )
        if coords.shape[0] > 0:
            dense[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        dense_levels[level] = dense
    return octree2tensor(dense_levels).float().contiguous()


def load_octree(
    sha256: str,
    octree_dir: str,
    base_resolution: int,
    num_levels: int,
) -> Tuple[str, Optional[torch.Tensor], Optional[str]]:
    try:
        path = os.path.join(octree_dir, f'{sha256}.pt')
        octree_info = torch.load(path, map_location='cpu', weights_only=True)
        tensor = octree_info_to_tensor(octree_info, base_resolution, num_levels)
        return sha256, tensor, None
    except Exception as exc:
        return sha256, None, str(exc)


def batched(items: Iterable[Tuple[str, torch.Tensor]], batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_latent(path: str, latent: np.ndarray) -> None:
    np.savez_compressed(path, z=latent)


def clear_cuda_error() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Dataset root containing metadata.csv')
    parser.add_argument('--octree_root', type=str, default=None,
                        help='Root containing octrees_<resolution>')
    parser.add_argument('--octree_latent_root', type=str, default=None,
                        help='Root used to save octree latents')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects below this aesthetic score')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Base octree resolution; defaults to the training config')
    parser.add_argument('--num_levels', type=int, default=None,
                        help='Number of octree levels; defaults to the training config')
    parser.add_argument('--model_root', type=str, required=True,
                        help='Root directory of trained models')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Octree VAE training directory under model_root')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint tag, for example step0010000')
    parser.add_argument('--latent_name', type=str, default=None,
                        help='Optional output latent directory name')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instance-list file or comma-separated sha256 values')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=min(8, os.cpu_count() or 1))
    opt = edict(vars(parser.parse_args()))

    if opt.rank < 0 or opt.rank >= opt.world_size:
        raise ValueError('rank must satisfy 0 <= rank < world_size')
    if opt.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if opt.num_workers <= 0:
        raise ValueError('num_workers must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to encode octree latents')

    opt.octree_root = opt.octree_root or opt.root
    opt.octree_latent_root = opt.octree_latent_root or opt.root
    _, encoder, resolution, num_levels = load_model_bundle(opt)
    latent_name = build_latent_name(opt, resolution)
    octree_dir = os.path.join(opt.octree_root, f'octrees_{resolution}')
    latent_dir = os.path.join(opt.octree_latent_root, 'octree_latents', latent_name)
    records_dir = os.path.join(latent_dir, 'new_records')

    if not os.path.isdir(octree_dir):
        raise FileNotFoundError(f'Octree directory not found: {octree_dir}')
    os.makedirs(records_dir, exist_ok=True)

    metadata_path = os.path.join(opt.root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'Metadata not found: {metadata_path}')
    metadata = pd.read_csv(metadata_path).set_index('sha256')
    metadata = combine_metadata(
        metadata,
        os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv'),
    )
    metadata = combine_metadata(metadata, os.path.join(octree_dir, 'metadata.csv'))
    metadata = combine_metadata(metadata, os.path.join(latent_dir, 'metadata.csv'))
    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            if 'aesthetic_score' not in metadata.columns:
                raise ValueError('aesthetic_score is required for aesthetic filtering')
            metadata = metadata[
                metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score
            ]
        if 'octreelized' in metadata.columns:
            metadata = metadata[metadata['octreelized'] == True]
    else:
        instances = load_instance_list(opt.instances)
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata.iloc[start:end]

    records = []
    pending_sha256s = []
    for sha256 in metadata['sha256'].values:
        output_path = os.path.join(latent_dir, f'{sha256}.npz')
        if os.path.exists(output_path):
            try:
                with np.load(output_path, allow_pickle=False) as pack:
                    latent = pack['z']
                    if latent.ndim != 4 or not np.isfinite(latent).all():
                        raise ValueError('Invalid latent array')
                records.append({
                    'sha256': sha256,
                    'octree_latent_encoded': True,
                    'octree_latent_channels': int(latent.shape[0]),
                    'octree_latent_resolution': int(latent.shape[-1]),
                })
                continue
            except Exception as exc:
                print(f'[Existing latent error] {sha256}: {exc}')
        pending_sha256s.append(sha256)

    print(f'Found {len(records)} existing latents')
    print(f'Processing {len(pending_sha256s)} objects...')

    loader = lambda sha256: load_octree(
        sha256,
        octree_dir,
        resolution,
        num_levels,
    )
    save_futures = []
    with ThreadPoolExecutor(max_workers=opt.num_workers) as loader_executor, \
            ThreadPoolExecutor(max_workers=opt.num_workers) as saver_executor:
        loaded = loader_executor.map(loader, pending_sha256s)

        def valid_loaded_items():
            for sha256, tensor, error in loaded:
                if tensor is None:
                    print(f'[Loader error] {sha256}: {error}')
                    continue
                yield sha256, tensor

        for batch in tqdm(
            batched(valid_loaded_items(), opt.batch_size),
            total=(len(pending_sha256s) + opt.batch_size - 1) // opt.batch_size,
            desc='Encoding octree latents',
        ):
            sha256s = [item[0] for item in batch]
            try:
                inputs = torch.stack([item[1] for item in batch], dim=0).cuda()
                latents = encoder(inputs, sample_posterior=False)
                torch.cuda.synchronize()
                if latents.ndim != 5 or not torch.isfinite(latents).all():
                    raise ValueError('Encoder produced an invalid latent tensor')

                latents = latents.detach().float().cpu().numpy()
                for sha256, latent in zip(sha256s, latents):
                    output_path = os.path.join(latent_dir, f'{sha256}.npz')
                    save_futures.append(
                        saver_executor.submit(save_latent, output_path, latent)
                    )
                    records.append({
                        'sha256': sha256,
                        'octree_latent_encoded': True,
                        'octree_latent_channels': int(latent.shape[0]),
                        'octree_latent_resolution': int(latent.shape[-1]),
                    })
            except Exception as exc:
                print(f'[Encoder error] {sha256s}: {exc}')
                clear_cuda_error()

        for future in save_futures:
            future.result()

    records = pd.DataFrame.from_records(
        records,
        columns=[
            'sha256',
            'octree_latent_encoded',
            'octree_latent_channels',
            'octree_latent_resolution',
        ],
    )
    records.to_csv(
        os.path.join(records_dir, f'part_{opt.rank}.csv'),
        index=False,
    )
