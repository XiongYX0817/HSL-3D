# Dataset Preparation Toolkit

This toolkit provides the data preparation pipeline for HSL-3D, including downloading assets, processing meshes, building O-Voxels and hierarchical octrees, rendering image conditions, and encoding the latents used for flow-model training.

### Step 1: Install Dependencies

Initialize the environment and install the required dependencies:

```bash
. ./data_toolkit/setup.sh
```

### Step 2: Initialize Metadata

Before processing, load the dataset metadata through its dataset adapter:

```bash
python data_toolkit/build_metadata.py <DATASET> --root <ROOT> [DATASET-SPECIFIC OPTIONS]
```

**Arguments:**

- `DATASET`: Name of the adapter under `datasets/`.
- `ROOT`: Root directory used to store the dataset and its metadata.

### Step 3: Download Data

Download the source 3D assets:

```bash
python data_toolkit/download.py <DATASET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

`RANK` and `WORLD_SIZE` can be used to split the work across multiple processes or nodes.

**Update Metadata:**

```bash
python data_toolkit/build_metadata.py <DATASET> --root <ROOT>
```

### Step 4: Process Meshes

Normalize the source assets and collect basic mesh statistics:

```bash
# Dump normalized meshes
python data_toolkit/dump_mesh.py <DATASET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]

# Compute vertex and face statistics
python data_toolkit/asset_stats.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Update Metadata:**

```bash
python data_toolkit/build_metadata.py <DATASET> --root <ROOT>
```

### Step 5: Build O-Voxels and Hierarchical Octrees

Convert the normalized meshes to dual-grid O-Voxels:

```bash
python data_toolkit/dual_grid.py <DATASET> --root <ROOT> \
    [--rank <RANK> --world_size <WORLD_SIZE>] \
    [--resolution <RESOLUTION>]
```

Build the hierarchical octrees used by the Stage I VAE and the Stage II octree frontier:

```bash
python data_toolkit/octreelize.py <DATASET> --root <ROOT> \
    [--rank <RANK> --world_size <WORLD_SIZE>] \
    [--resolution <OCTREE_RESOLUTION>]
```



**Example:**

```bash
python data_toolkit/dual_grid.py <DATASET> --root <ROOT> --resolution 1024
python data_toolkit/octreelize.py <DATASET> --root <ROOT> --resolution 16
```

**Update Metadata:**

```bash
python data_toolkit/build_metadata.py <DATASET> --root <ROOT>
```



### Step 6: Encode Latents

After training the corresponding VAEs, encode their latents for flow-model training.

Encode Stage I octree latents:

```bash
python data_toolkit/encode_octree_latent.py \
    --root <ROOT> \
    --model_root <MODEL_ROOT> \
    --model_dir <OCTREE_VAE_MODEL_DIR> \
    --ckpt <CKPT> \
    [--rank <RANK> --world_size <WORLD_SIZE>]
```

Encode Stage II multiscale shape latents:

```bash
python data_toolkit/encode_multiscale_shape_latent.py \
    --root <ROOT> \
    --resolution <RESOLUTION> \
    --model_root <MODEL_ROOT> \
    --model_dir <SHAPE_VAE_MODEL_DIR> \
    --ckpt <CKPT> \
    [--rank <RANK> --world_size <WORLD_SIZE>]
```

The exported latents use zero-based levels: level 0, 1, and 2 correspond to grid resolutions 16, 32, and 64, respectively.



**Arguments:**

- `MODEL_ROOT`: Root directory containing the VAE training runs.
- `OCTREE_VAE_MODEL_DIR` / `SHAPE_VAE_MODEL_DIR`: VAE run directory under `MODEL_ROOT`.
- `CKPT`: Encoder checkpoint tag, such as `step1000000`.
- `RESOLUTION`: Dual-grid resolution used by the Stage II VAE.


**Update Metadata:**

```bash
python data_toolkit/build_metadata.py <DATASET> --root <ROOT>
```

### Step 7: Render Image Conditions

Render multi-view images for training the image-conditioned flow models:

```bash
python data_toolkit/render_cond.py <DATASET> --root <ROOT> \
    [--num_cond_views <NUM_VIEWS>] \
    [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Arguments:**

- `NUM_VIEWS`: Number of condition views rendered for each asset. The default is `16`.

**Final Metadata Update:**

```bash
python data_toolkit/build_metadata.py <DATASET> --root <ROOT>
```
