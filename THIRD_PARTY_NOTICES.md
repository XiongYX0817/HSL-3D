# Third-Party Notices

The repository source code is licensed under the root `LICENSE` file except for components identified below. Third-party components and dependencies are not relicensed under the repository's MIT License.

This file focuses on source code bundled with the repository, Git submodules, and dependencies installed directly from upstream source repositories. Normal Python package dependencies installed from package indexes retain the license terms shipped by their respective distributions.

## Upstream project

### Microsoft TRELLIS.2 and O-Voxel

- Source: https://github.com/microsoft/TRELLIS.2
- License: MIT
- Copyright: Microsoft Corporation
- Use in this repository: HSL-3D is derived from TRELLIS.2. The `o-voxel/` source was distributed as part of TRELLIS.2 under its top-level MIT License.
- License text: `LICENSE`

The original copyright and license notice must be retained in copies or substantial portions of the derived source.

## Component distributed in this repository

### BlenderUSDZ

- Bundled path: `data_toolkit/blender_script/io_scene_usdz.zip`
- Upstream source: https://github.com/robmcrosby/BlenderUSDZ
- Author: Robert Crosby
- Bundled add-on version: 0.0.6
- Bundled archive SHA-256:
  `EC07AB6125FE0A021ED08C64169ECEDA126330401ABA3D494D5203D26AC4B093`
- License: GNU General Public License version 3
- License text: `licenses/BlenderUSDZ-GPL-3.0.txt`

The archive contains the add-on's Python source. It remains licensed under GPL-3.0 and is not covered by the repository's MIT License.

## Git submodule

### Eigen

- Path: `o-voxel/third_party/eigen`
- Source: https://gitlab.com/libeigen/eigen.git
- Primary license: Mozilla Public License 2.0

Eigen contains its own license and third-party notice files. Those files must remain present when the submodule is initialized or redistributed. The exact submodule revision should be pinned before release.

## External source dependencies

The following projects are fetched or installed separately and are not included in this repository unless stated otherwise.

### FlexGEMM

- Source: https://github.com/JeffreyXiang/FlexGEMM
- License: MIT
- Role: sparse convolution and 3D grid sampling backend

### CuMesh

- Source: https://github.com/JeffreyXiang/CuMesh
- Top-level license: MIT
- Role: CUDA mesh processing and post-processing

CuMesh is installed recursively and includes additional components. In particular, its `third_party/cubvh` submodule points to the `trellis.2` branch of https://github.com/JeffreyXiang/cubvh. That fork contains both an MIT license and an NVIDIA Source Code License covering NVIDIA-derived material. The NVIDIA-covered material is restricted to non-commercial research or evaluation use. CuMesh also contains xatlas code under the MIT License.

### nvdiffrast

- Source: https://github.com/NVlabs/nvdiffrast/tree/v0.4.0
- License: NVIDIA Source Code License (1-Way Commercial)
- Role: differentiable rasterization

The upstream license restricts use to non-commercial research or evaluation, except for NVIDIA and its affiliates. Redistribution requires retaining the complete upstream license and notices.

### nvdiffrec render utilities

- Source: https://github.com/JeffreyXiang/nvdiffrec/tree/renderutils
- License: NVIDIA Source Code License for nvdiffrec
- Role: environment lighting and PBR rendering utilities

The upstream license restricts use to non-commercial research or evaluation, except for NVIDIA and its affiliates. The actual source installed by `setup.sh` is the `renderutils` branch of this fork.

### DINOv3

- Code source: https://github.com/facebookresearch/dinov3
- Model source: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- License: DINOv3 License
- Role: image-conditioning features

DINOv3 uses a custom license that applies to its code and model materials. Users obtain the model separately from the upstream provider and must accept and comply with that license, including its redistribution, attribution, trade-control, and prohibited-use terms.

### BRIA RMBG-2.0

- Model source: https://huggingface.co/briaai/RMBG-2.0
- License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
- Role: optional input-image background removal

The model weights are obtained separately from the upstream provider and are
licensed for non-commercial use. Commercial use requires a separate agreement
with BRIA. The pipeline can skip this dependency when the input image already
contains a usable alpha channel or when image preprocessing is disabled.

### utils3d

- Source: https://github.com/EasternJournalist/utils3d
- Revision used by `setup.sh`: `9a4eb15e4021b67b12c460c7057d642626897ec8`
- License: MIT
- Role: camera, projection, mesh, and rendering utilities

### FlashAttention

- Source: https://github.com/Dao-AILab/flash-attention
- Version requested by `setup.sh`: 2.7.3
- License: BSD 3-Clause
- Role: attention backend

### Optional sparse and attention backends

- xFormers: https://github.com/facebookresearch/xformers — BSD 3-Clause
- spconv: https://github.com/traveller59/spconv — Apache License 2.0
- TorchSparse: https://github.com/mit-han-lab/torchsparse — MIT

These projects are supported as optional backends by the source code. They retain their upstream license terms if installed.

## Release practice

Before a public release:

1. Pin source dependencies and Git submodules to reviewed revisions.
2. Preserve every license and notice shipped by initialized submodules.
3. Do not vendor external dependency source or binaries without also carrying the license and notice files required by that dependency.
4. Re-audit this file whenever installation scripts or bundled third-party files change.
