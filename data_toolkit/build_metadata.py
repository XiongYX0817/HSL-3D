import os
import shutil
import sys
import time
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict


def update_metadata(path, opt):
    if not os.path.exists(path):
        return None
    timestamp = str(int(time.time()))
    os.makedirs(os.path.join(path, 'merged_records'), exist_ok=True)
    os.makedirs(os.path.join(path, 'new_records'), exist_ok=True)
    if opt.from_merged_records:
        records_dir = os.path.join(path, 'merged_records')
        df_files = [f for f in os.listdir(records_dir) if f.endswith('.csv')]
        if opt.record_start is not None:
            df_files = [f for f in df_files if int(f.split('_')[0]) >= opt.record_start]
    else:
        records_dir = os.path.join(path, 'new_records')
        df_files = [f for f in os.listdir(records_dir) if f.startswith('part_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(records_dir, f)))
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    if len(df_parts) > 0:
        if os.path.exists(os.path.join(path, 'metadata.csv')):
            metadata = pd.read_csv(os.path.join(path, 'metadata.csv'))
        else:
            columns = df_parts[0].columns
            metadata = pd.DataFrame(columns=columns)
        metadata.set_index('sha256', inplace=True)
        for df_part in df_parts:
            if 'sha256' in df_part.columns:
                df_part.set_index('sha256', inplace=True)
                metadata = df_part.combine_first(metadata)
        metadata.to_csv(os.path.join(path, 'metadata.csv'))
        if not opt.from_merged_records:
            for f in df_files:
                shutil.move(
                    os.path.join(records_dir, f),
                    os.path.join(path, 'merged_records', f'{timestamp}_{f}'),
                )
        return metadata
    else:
        if os.path.exists(os.path.join(path, 'metadata.csv')):
            return pd.read_csv(os.path.join(path, 'metadata.csv'))
    return None


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to save the downloaded files')
    parser.add_argument('--render_cond_root', type=str, default=None,
                        help='Directory to save the render condition files')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to save the mesh files')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory to save the dual grid files')
    parser.add_argument('--octree_root', type=str, default=None,
                        help='Directory containing hierarchical octrees')
    parser.add_argument('--octree_latent_root', type=str, default=None,
                        help='Directory containing octree latent files')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory containing multiscale shape latent files')
    parser.add_argument('--from_merged_records', action='store_true',
                        help='Build metadata from merged records')
    parser.add_argument('--record_start', type=int)
    dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.render_cond_root = opt.render_cond_root or opt.root
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.octree_root = opt.octree_root or opt.root
    opt.octree_latent_root = opt.octree_latent_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root

    os.makedirs(opt.root, exist_ok=True)
    
    # get file list
    if os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        print('Loading previous metadata...')
        metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv'))
    else:
        metadata = dataset_utils.get_metadata(**opt)
        metadata.to_csv(os.path.join(opt.root, 'metadata.csv'), index=False)
    
    # merge downloaded
    downloaded_metadata = update_metadata(os.path.join(opt.download_root, 'raw'), opt)

    # merge aesthetic scores
    aesthetic_score_metadata = update_metadata(os.path.join(opt.root, 'aesthetic_scores'), opt)
    
    # merge render conditions
    render_cond_metadata = update_metadata(os.path.join(opt.render_cond_root, 'renders_cond'), opt)

    # merge mesh dumped
    mesh_dumped_metadata = update_metadata(os.path.join(opt.mesh_dump_root, 'mesh_dumps'), opt)
        
    # merge asset stats
    asset_stats_metadata = update_metadata(os.path.join(opt.root, 'asset_stats'), opt)
        
    # merge dual grid
    dual_grid_resolutions = []
    for dir in os.listdir(opt.dual_grid_root):
        if os.path.isdir(os.path.join(opt.dual_grid_root, dir)) and dir.startswith('dual_grid_'):
            dual_grid_resolutions.append(int(dir.split('_')[-1]))
    dual_grid_metadata = {}
    for res in dual_grid_resolutions:
        dual_grid_metadata[res] = update_metadata(os.path.join(opt.dual_grid_root, f'dual_grid_{res}'), opt)
    
    # merge octree
    octree_resolutions = []
    for dir in os.listdir(opt.octree_root):
        if os.path.isdir(os.path.join(opt.octree_root, dir)) and dir.startswith('octrees_'):
            suff = dir.split('_')[-1]
            if suff not in ["sp"]:
                octree_resolutions.append(int(suff))
    octree_metadata = {}
    for res in octree_resolutions:
        octree_metadata[res] = update_metadata(os.path.join(opt.octree_root, f'octrees_{res}'), opt)

    # merge octree latents
    octree_latent_models = []
    octree_latent_dir = os.path.join(opt.octree_latent_root, 'octree_latents')
    if os.path.exists(octree_latent_dir):
        octree_latent_models = [
            name
            for name in os.listdir(octree_latent_dir)
            if os.path.isdir(os.path.join(octree_latent_dir, name))
        ]
    octree_latent_metadata = {}
    for model in octree_latent_models:
        octree_latent_metadata[model] = update_metadata(
            os.path.join(octree_latent_dir, model),
            opt,
        )

    # merge shape latents
    shape_latent_models = []
    if os.path.exists(os.path.join(opt.shape_latent_root, 'shape_latents')):
        shape_latent_models = os.listdir(os.path.join(opt.shape_latent_root, 'shape_latents'))
    shape_latent_metadata = {}
    for model in shape_latent_models:
        shape_latent_metadata[model] = update_metadata(os.path.join(opt.shape_latent_root, f'shape_latents/{model}'), opt)

    # statistics
    num_downloaded = downloaded_metadata['local_path'].count() if downloaded_metadata is not None else 0
    with open(os.path.join(opt.root, 'statistics.txt'), 'w') as f:
        f.write('Statistics:\n')
        f.write(f'  - Number of assets: {len(metadata)}\n')
        f.write(f'  - Number of assets downloaded: {num_downloaded}\n')
        if aesthetic_score_metadata is not None:
            f.write(f'  - Number of assets with aesthetic scores: {aesthetic_score_metadata["aesthetic_score"].count()}\n')
        if render_cond_metadata is not None:
            f.write(f'  - Number of assets with render conditions: {render_cond_metadata["cond_rendered"].count()}\n')
        if mesh_dumped_metadata is not None:
            f.write(f'  - Number of assets with mesh dumped: {mesh_dumped_metadata["mesh_dumped"].sum()}\n')
        if asset_stats_metadata is not None:
            f.write(f'  - Number of assets with asset stats: {len(asset_stats_metadata)}\n')
        if len(dual_grid_resolutions) != 0:
            f.write(f'  - Number of assets with dual grid:\n')
            for res in dual_grid_resolutions:
                if dual_grid_metadata[res] is not None:
                    f.write(f'    - {res}: {dual_grid_metadata[res]["dual_grid_converted"].sum()}\n')
        if len(octree_resolutions) != 0:
            f.write(f'  - Number of assets with octree:\n')
            for res in octree_resolutions:
                if octree_metadata[res] is not None:
                    f.write(f'    - {res}: {octree_metadata[res]["octreelized"].sum()}\n')
        if len(octree_latent_models) != 0:
            f.write('  - Number of assets with octree latents:\n')
            for model in octree_latent_models:
                if octree_latent_metadata[model] is not None:
                    f.write(
                        f'    - {model}: '
                        f'{octree_latent_metadata[model]["octree_latent_encoded"].sum()}\n'
                    )
        if len(shape_latent_models) != 0:
            f.write(f'  - Number of assets with shape latents:\n')
            for model in shape_latent_models:
                if shape_latent_metadata[model] is not None:
                    f.write(f'    - {model}: {shape_latent_metadata[model]["shape_latent_encoded"].sum()}\n')
        
    with open(os.path.join(opt.root, 'statistics.txt'), 'r') as f:
        print(f.read())
