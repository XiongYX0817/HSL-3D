import os
import argparse
import pickle
from tqdm import tqdm
import pandas as pd
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to save the mesh dumps')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root

    os.makedirs(os.path.join(opt.root, 'asset_stats', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'asset_stats', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'asset_stats','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps','metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if 'num_faces' in metadata.columns:
            metadata = metadata[metadata['num_faces'].isnull()]
        metadata = metadata[metadata['mesh_dumped'] == True]
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
       
    print(f'Processing {len(metadata)} objects...')

    # process objects
    records = []
    with ThreadPoolExecutor(max_workers=opt.max_workers or os.cpu_count()) as executor, \
         tqdm(total=len(metadata), desc='Processing objects') as pbar:
        def worker(metadatum):
            sha256 = metadatum['sha256']
            try:
                with open(os.path.join(opt.mesh_dump_root, 'mesh_dumps', f'{sha256}.pickle'), 'rb') as f:
                    dump = pickle.load(f)

                num_faces = 0
                num_vertices = 0
                for obj in dump['objects']:
                    if obj['vertices'].size == 0 or obj['faces'].size == 0:
                        continue
                    num_faces += obj['faces'].shape[0]
                    num_vertices += obj['vertices'].shape[0]

                records.append({
                    'sha256': sha256,
                    'num_faces': num_faces,
                    'num_vertices': num_vertices,
                })
                pbar.update()
            except Exception as e:
                print(f'Error processing {sha256}: {e}')
                pbar.update()                

        for metadatum in metadata.to_dict('records'):
            executor.submit(worker, metadatum)

        executor.shutdown(wait=True)

    # save records
    records = pd.DataFrame.from_records(
        records,
        columns=['sha256', 'num_faces', 'num_vertices'],
    )
    records.to_csv(os.path.join(opt.root, 'asset_stats', 'new_records', f'part_{opt.rank}.csv'), index=False)
