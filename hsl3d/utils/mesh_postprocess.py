from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch

import cumesh
from hsl3d.representations import Mesh
from hsl3d.utils.mesh_utils import read_ply, write_ply


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def copy_mesh(mesh: Mesh) -> Mesh:
    vertex_attrs = None if mesh.vertex_attrs is None else mesh.vertex_attrs.detach().clone()
    return Mesh(
        vertices=mesh.vertices.detach().clone(),
        faces=mesh.faces.detach().clone(),
        vertex_attrs=vertex_attrs,
    )


def postprocess_shape_mesh(
    mesh: Mesh,
    *,
    decimation_target: int,
    max_hole_perimeter: float,
    cleanup_small_component_threshold: float,
) -> Mesh:
    post = copy_mesh(mesh)
    if post.vertices.numel() == 0 or post.faces.numel() == 0:
        return post

    cmesh = cumesh.CuMesh()
    cmesh.init(post.vertices.cuda(), post.faces.cuda())
    cmesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
    cmesh.simplify(int(decimation_target) * 3, verbose=False)
    cmesh.remove_duplicate_faces()
    cmesh.repair_non_manifold_edges()
    cmesh.remove_small_connected_components(cleanup_small_component_threshold)
    cmesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
    cmesh.simplify(int(decimation_target), verbose=False)

    new_vertices, new_faces = cmesh.read()
    return Mesh(new_vertices.to(post.device), new_faces.to(post.device))


def triangulate_face(face: Iterable[int]) -> List[List[int]]:
    face = list(face)
    if len(face) < 3:
        return []
    if len(face) == 3:
        return [face]
    return [[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)]


def parse_obj_vertex_index(token: str, num_vertices: int) -> int:
    vertex_token = token.split("/")[0]
    if not vertex_token:
        raise ValueError(f"OBJ face token is missing a vertex index: {token!r}")

    index = int(vertex_token)
    if index > 0:
        return index - 1
    if index < 0:
        return num_vertices + index
    raise ValueError("OBJ indices are 1-based; index 0 is invalid.")


def load_obj_mesh(path: str | Path) -> Mesh:
    vertices: List[List[float]] = []
    triangles: List[List[int]] = []
    path = Path(path).expanduser().resolve()

    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for line_no, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if not parts:
                continue

            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"OBJ vertex line has fewer than 3 coordinates at {path}:{line_no}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if parts[0] == "f":
                if len(parts) < 4:
                    raise ValueError(f"OBJ face line has fewer than 3 vertices at {path}:{line_no}")
                face = [parse_obj_vertex_index(token, len(vertices)) for token in parts[1:]]
                triangles.extend(triangulate_face(face))

    if not vertices:
        raise ValueError(f"No vertices found in OBJ file: {path}")

    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    faces_tensor = (
        torch.tensor(triangles, dtype=torch.int32)
        if triangles
        else torch.empty((0, 3), dtype=torch.int32)
    )
    return Mesh(vertices_tensor, faces_tensor)


def load_ply_mesh(path: str | Path) -> Mesh:
    vertices, triangles, quads = read_ply(str(Path(path).expanduser().resolve()))
    vertices_tensor = torch.from_numpy(vertices).to(dtype=torch.float32)
    triangles_tensor = torch.from_numpy(triangles).to(dtype=torch.int32)

    quad_triangles: List[List[int]] = []
    for quad in quads.tolist():
        quad_triangles.extend(triangulate_face(quad))

    if quad_triangles:
        quad_tensor = torch.tensor(quad_triangles, dtype=torch.int32)
        faces_tensor = torch.cat([triangles_tensor, quad_tensor], dim=0)
    else:
        faces_tensor = triangles_tensor

    return Mesh(vertices_tensor, faces_tensor)


def load_mesh(path: str | Path) -> Mesh:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj_mesh(path)
    if suffix == ".ply":
        return load_ply_mesh(path)
    raise ValueError(f"Unsupported mesh format: {suffix}. Expected .obj or .ply")


def save_obj_mesh(path: str | Path, mesh: Mesh) -> None:
    path = Path(path).expanduser().resolve()
    ensure_parent_dir(path)
    vertices = mesh.vertices.detach().cpu().float()
    faces = mesh.faces.detach().cpu().to(torch.int64)

    with open(path, "w", encoding="utf-8") as fp:
        for vertex in vertices.tolist():
            fp.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces.tolist():
            fp.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_ply_mesh(path: str | Path, mesh: Mesh, *, ascii_ply: bool) -> None:
    path = Path(path).expanduser().resolve()
    ensure_parent_dir(path)
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    quads = torch.empty((0, 4), dtype=torch.int32).numpy()
    write_ply(str(path), vertices, faces, quads, ascii=ascii_ply)


def save_mesh(path: str | Path, mesh: Mesh, *, ascii_ply: bool = False) -> None:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".obj":
        save_obj_mesh(path, mesh)
        return
    if suffix == ".ply":
        save_ply_mesh(path, mesh, ascii_ply=ascii_ply)
        return
    raise ValueError(f"Unsupported output mesh format: {suffix}. Expected .obj or .ply")


def default_output_path(input_path: str | Path) -> str:
    src = Path(input_path).expanduser().resolve()
    return str(src.with_name(f"{src.stem}_postprocessed{src.suffix}"))


def mesh_stats(mesh: Mesh) -> dict:
    return {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
    }
