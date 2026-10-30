import torch


def face_areas_normals(V, F):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = 0.5 * torch.norm(cross, dim=1)
    normals = cross / (2 * areas[:, None] + 1e-12)

    return areas, normals

def vertex_normals(V, F, face_normals, face_areas):
    vnormals = torch.zeros_like(V)

    weighted = face_normals * face_areas[:, None]

    for i in range(3):
        vnormals.index_add_(0, F[:, i], weighted)

    vnormals = vnormals / (torch.norm(vnormals, dim=1, keepdim=True) + 1e-12)
    return vnormals

# def sample_faces(face_areas, n_samples):
#     probs = face_areas / face_areas.sum()
#     face_idx = torch.multinomial(probs, n_samples, replacement=True)
#     return face_idx

def sample_faces(face_areas, n_samples):
    face_areas = face_areas.clone()
    face_areas[~torch.isfinite(face_areas)] = 0
    face_areas = torch.clamp(face_areas, min=1e-6)

    total = face_areas.sum()

    if total <= 0:
        # fallback：均匀分布
        probs = torch.ones_like(face_areas) / face_areas.numel()
    else:
        probs = face_areas / total

    return torch.multinomial(probs, n_samples, replacement=True)

def sample_barycentric(n_samples, device):
    u = torch.rand(n_samples, device=device)
    v = torch.rand(n_samples, device=device)

    mask = (u + v) > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]

    w = 1 - u - v
    return torch.stack([w, u, v], dim=1)

def sample_points_and_normals(V, F, vnormals, face_areas, n_samples):
    device = V.device

    # 1. 采样三角形
    face_idx = sample_faces(face_areas, n_samples)
    faces = F[face_idx]

    # 2. 重心坐标
    bary = sample_barycentric(n_samples, device)

    # 3. 采样点
    verts = V[faces]                       # (N,3,3)
    points = (verts * bary[:, :, None]).sum(dim=1)

    # 4. 插值法线
    normals = (vnormals[faces] * bary[:, :, None]).sum(dim=1)
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-12)

    return points, normals
