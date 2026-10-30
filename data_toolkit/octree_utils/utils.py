import torch
import numpy as np
from plyfile import PlyData, PlyElement


def normalize_to_unit_cube(vertices):
    """
    Normalize mesh to [-1, 1]^3:
    - keep aspect ratio
    - mesh center at origin
    - longest axis scaled to [-1, 1]

    vertices: (N, 3) torch tensor
    return:   (N, 3) torch tensor
    """
    # Axis-aligned bounding box
    min_vals = vertices.min(dim=0)[0]
    max_vals = vertices.max(dim=0)[0]

    # Mesh center
    center = (min_vals + max_vals) * 0.5

    # Longest extent
    extent = max_vals - min_vals
    scale = extent.max() * 0.5

    # Normalize
    normalized = (vertices - center) / scale

    return normalized

def scalar_to_color_heat(values, eps=1e-8):
    lo = torch.quantile(values, 0.01)
    hi = torch.quantile(values, 0.99)
    values = values.clamp(lo, hi)
    t = (values - values.min()) / (values.max() - values.min() + eps)
    r = torch.clamp(1.5 * t - 0.5, 0, 1)
    g = torch.clamp(1.5 - torch.abs(2*t - 1.0), 0, 1)
    b = torch.clamp(1.0 - 1.5 * t, 0, 1)
    return torch.stack([r, g, b], dim=1)

def complexity_to_rgb(complexity):
    c = complexity - complexity.min()
    c = c / (c.max() + 1e-8)
    r = torch.ones_like(c)
    g = 1.0 - c
    b = 1.0 - c
    return torch.stack([r, g, b], dim=1)

def save_colored_obj(verts, faces, colors, path="test.obj"):

    # ---------- to numpy ----------
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    verts = verts.astype(np.float32)
    colors = colors.astype(np.float32)
    faces = faces.astype(np.int64)

    # ---------- color range handling ----------
    if colors.max() > 1.0:
        colors = colors / 255.0

    # ---------- write file ----------
    with open(path, "w") as f:
        # vertices with colors
        for v, c in zip(verts, colors):
            f.write(
                f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
            )

        # faces (OBJ is 1-based)
        for face in faces:
            f.write(
                f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
            )

def normals_to_rgb(normals: torch.Tensor):
    """
    normals: (N, 3), torch float32, range [-1, 1]
    return: (N, 3), uint8, range [0, 255]
    """
    rgb = (normals + 1.0) * 0.5 * 255.0
    rgb = torch.clamp(rgb, 0, 255).byte()
    return rgb

def save_pointcloud_ply(points: torch.Tensor, colors: torch.Tensor, filename: str):

    pts = points.cpu().numpy().astype(np.float32)
    colors = colors.cpu().numpy().astype(np.uint8)

    n = pts.shape[0]

    vertex_data = np.empty(
        n,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
    )

    vertex_data["x"] = pts[:, 0]
    vertex_data["y"] = pts[:, 1]
    vertex_data["z"] = pts[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=False).write(filename)

UNIT_CUBE_VERTICES = torch.tensor([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
], dtype=torch.float32)

CUBE_FACES = [
    (1, 2, 3), (1, 3, 4),
    (5, 6, 7), (5, 7, 8),
    (1, 2, 6), (1, 6, 5),
    (2, 3, 7), (2, 7, 6),
    (3, 4, 8), (3, 8, 7),
    (4, 1, 5), (4, 5, 8),
]

def export_voxels_to_obj(coords, colors=None, N=32, filename="voxels.obj"):

    coords = coords.cpu()
    K = coords.shape[0]

    colors = colors.cpu() if colors is not None else np.array([[1.0, 0.0, 0.0] for _ in range(K)])

    voxel_size = 2.0 / N

    centers = (coords.float() + 0.5) / N * 2.0 - 1.0

    cube_v = UNIT_CUBE_VERTICES * voxel_size * 0.99

    obj_lines = []

    v_offset = 0
    for i in range(K):
        r, g, b = colors[i].tolist()
        verts = cube_v + centers[i]
        for v in verts:
            obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}")
        for f in CUBE_FACES:
            a, b_, c = f
            obj_lines.append(f"f {a+v_offset} {b_+v_offset} {c+v_offset}")
        v_offset += 8

    with open(filename, "w") as f:
        f.write("\n".join(obj_lines))

def export_octree_to_obj(results, N=32, filename="voxels.obj"):

    COLORS = [
        [1.0, 0.8, 0.8],
        [1.0, 0.4, 0.4],
        [1.0, 0.0, 0.0]
    ]

    v_offset = 0
    obj_lines_v = []
    obj_lines_f = []

    levels = len(results.keys())
    for level in range(levels):

        # coords = results[level][0].cpu()
        coords = results[level].cpu()

        K = coords.shape[0]
        voxel_size = 2.0 / N

        centers = (coords.float() + 0.5) / N * 2.0 - 1.0

        cube_v = UNIT_CUBE_VERTICES * voxel_size * 0.98

        for i in range(K):
            r, g, b = COLORS[level]
            verts = cube_v + centers[i]
            for v in verts:
                obj_lines_v.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}")
            for f in CUBE_FACES:
                a, b_, c = f
                obj_lines_f.append(f"f {a+v_offset} {b_+v_offset} {c+v_offset}")
            v_offset += 8
        
        N *= 2

    with open(filename, "w") as f:
        f.write("\n".join(obj_lines_v + obj_lines_f))
