import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps

from hsl3d.pipelines import HSL3DImageTo3DPipeline
from hsl3d.representations import Mesh
from hsl3d.utils import render_utils
from hsl3d.utils.mesh_postprocess import save_mesh


def export_geometry_glb(path: Path, mesh: Mesh) -> None:
    """Export the generated Z-up geometry as a Y-up, material-free GLB."""
    vertices = mesh.vertices.detach().cpu().numpy().astype(np.float32, copy=True)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int64, copy=False)
    vertices = np.stack(
        [vertices[:, 0], vertices[:, 2], -vertices[:, 1]],
        axis=1,
    )
    glb_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    glb_mesh.export(str(path))


def save_normal_video(
    path: Path,
    mesh: Mesh,
    *,
    resolution: int,
    num_frames: int,
    fps: int,
    face_limit: int,
) -> None:
    render_mesh = mesh.to(mesh.device)
    if render_mesh.faces.shape[0] > face_limit:
        render_mesh.simplify(face_limit, verbose=False)
    frames = render_utils.render_video(
        render_mesh,
        resolution=resolution,
        num_frames=num_frames,
        return_types=["normal"],
    )["normal"]
    imageio.mimsave(path, frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate geometry from an image with the HSL-3D two-stage pipeline."
    )
    parser.add_argument("--model_path", required=True, help="Local model-bundle directory or model repository.")
    parser.add_argument("--config_file", default="pipeline.json", help="Pipeline config filename inside the model bundle.")
    parser.add_argument("--image_path", required=True, help="Input image path.")
    parser.add_argument("--output_dir", default="outputs/example", help="Directory for PLY, GLB, and MP4 outputs.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device used for inference and rendering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--no_preprocess_image", action="store_true")
    parser.add_argument("--video_resolution", type=int, default=1024)
    parser.add_argument("--video_num_frames", type=int, default=120)
    parser.add_argument("--video_fps", type=int, default=15)
    parser.add_argument("--video_face_limit", type=int, default=16777216)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for pipeline inference and rendering")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("example.py currently supports CUDA devices only")
    if device.index is not None:
        torch.cuda.set_device(device.index)

    pipeline = HSL3DImageTo3DPipeline.from_pretrained(
        args.model_path,
        config_file=args.config_file,
    )
    pipeline.to(device)

    image = ImageOps.exif_transpose(Image.open(args.image_path))
    meshes = pipeline.run(
        image,
        num_samples=args.num_samples,
        seed=args.seed,
        preprocess_image=not args.no_preprocess_image,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, mesh in enumerate(meshes):
        stem = f"sample_{index:02d}" if len(meshes) > 1 else "sample"
        save_mesh(output_dir / f"{stem}.ply", mesh)
        export_geometry_glb(output_dir / f"{stem}.glb", mesh)
        save_normal_video(
            output_dir / f"{stem}.mp4",
            mesh,
            resolution=args.video_resolution,
            num_frames=args.video_num_frames,
            fps=args.video_fps,
            face_limit=args.video_face_limit,
        )


if __name__ == "__main__":
    main()
