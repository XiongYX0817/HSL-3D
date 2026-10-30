import torch

def voxelize(points, N=32, device='cuda'):
    # points = torch.clamp((points+1)*0.5, 0, 1.0-1e-6)
    points = torch.clamp(points+0.5, 0, 1.0-1e-6)
    voxel_indices = torch.floor(points * N).long().to(device)
    return voxel_indices

def flatten_voxel_index(voxel_indices, N):
    return voxel_indices[:, 0] * (N * N) + voxel_indices[:, 1] * N + voxel_indices[:, 2]

def compute_normal_variance(normals, voxel_id, N):

    device = normals.device
    P = normals.shape[0]
    V = N * N * N

    # ---- 1. accumulate normal sum and count per voxel
    voxel_normal_sum = torch.zeros((V, 3), device=device)
    voxel_count = torch.zeros((V, 1), device=device)
    voxel_normal_sum.scatter_add_(0, voxel_id[:, None].expand(-1, 3), normals)
    voxel_count.scatter_add_(0, voxel_id[:, None], torch.ones((P, 1), device=device))

    # ---- 2. voxel mean normal
    voxel_mean = voxel_normal_sum / (voxel_count + 1e-6)

    # ---- 3. per-point squared residual
    diff = normals - voxel_mean[voxel_id]                  # (P,3)
    sq_norm = torch.sum(diff * diff, dim=1, keepdim=True)  # (P,1)

    # ---- 4. accumulate variance
    voxel_var_sum = torch.zeros((V, 1), device=device)
    voxel_var_sum.scatter_add_(0, voxel_id[:, None], sq_norm)
    voxel_variance = voxel_var_sum / (voxel_count + 1e-6)  # (V,1)

    # ---- 5. keep occupied voxels only
    mask = voxel_count[:, 0] > 0
    occ_voxel_id = torch.nonzero(mask, as_tuple=False).squeeze(1)
    occ_variance = voxel_variance[mask, 0]  # (K,)

    # ---- 6. voxel id -> (x,y,z)
    x = occ_voxel_id // (N * N)
    y = (occ_voxel_id % (N * N)) // N
    z = occ_voxel_id % N

    coords = torch.stack([x, y, z], dim=1)  # (K,3)

    return coords, occ_variance

def generate_finest_voxels(points, N=128, device='cuda'):
    voxel_indices = voxelize(points, N, device)
    return torch.unique(voxel_indices, dim=0)

def generate_split_mask(points, normals, level=3, N=32, device='cuda', _currentlevel=0, threshold=0.5):
    results = dict()
    voxel_indices = voxelize(points, N, device)
    p_voxel_id = flatten_voxel_index(voxel_indices, N)
    coords, complexity = compute_normal_variance(normals, p_voxel_id, N)
    voxel_mask = complexity > threshold
    active_voxel_ids = flatten_voxel_index(coords, N)
    if active_voxel_ids.shape[0] > 0:
        point_mask = torch.isin(p_voxel_id, active_voxel_ids[voxel_mask])
        if _currentlevel+2 < level:
            results = generate_split_mask(points[point_mask], normals[point_mask], level, 2*N, device, _currentlevel+1, threshold)
    results[_currentlevel] = coords[voxel_mask]
    return results

def process_pointcloud(points, normals, level=3, res=16, threshold=0.5, gt_finest_voxel=None, device='cuda'):
    results = generate_split_mask(points, normals, level=level, N=res, device=device, _currentlevel=0, threshold=threshold)
    if gt_finest_voxel is None:
        results[level-1] = generate_finest_voxels(points, N=res*2**(level-1), device=device)
    else:
        results[level-1] = gt_finest_voxel.to(device)
    return results

def downsample(coords, N0, N1):
    return (coords / N0 * N1).long()

def isin(A, B, factor=10_000):
    A_key = A[:, 0]*factor**2 + A[:, 1]*factor + A[:, 2]
    B_key = B[:, 0]*factor**2 + B[:, 1]*factor + B[:, 2]
    return torch.isin(A_key, B_key)

def diff(A, B, factor=10_000):
    return A[~isin(A, B, factor)]

def rebuild_octree(octree_info, N=16, level=3):
    octree = dict()
    finest_coords = octree_info[level-1]
    N_current, N_finest = N, N * 2**(level-1)
    for i in range(level-1):
        split_coords = octree_info[i]
        cur_coords = downsample(finest_coords, N_finest, N_current)
        octree[i] = diff(torch.unique(cur_coords, dim=0), split_coords, factor=N_finest)
        finest_mask = ~isin(cur_coords, octree[i])
        finest_coords = finest_coords[finest_mask.bool()]
        N_current *= 2
    octree[level-1] = finest_coords
    return octree
