# Particle Imaging Models (pimm)
## Foundation model research for particle imaging detectors

A codebase for perception research for time projection chambers (TPCs), with a focus on liquid argon TPCs, built on the [Pointcept](https://github.com/Pointcept/Pointcept) training and inference framework.

This repository currently deals with _3D charge clouds_ only, with plans to incorporate 2D
images (e.g., wireplane waveforms) and other modalities in the near future.

## Overview

**pimm** adapts 3D point cloud methods for event reconstruction in LArTPC detectors. This repository provides:

- **Self-supervised pre-training**: discriminative pre-training ([Sonata](https://arxiv.org/abs/2503.16429))
- **Panoptic segmentation** (PointGroup, Panda Detector)models for particle instance and semantic segmentation
- **Semantic segmentation** for motif-level (track, shower, ...) per-pixel segmentation.

In sum, **pimm** integrates the following works:  
**Backbone**: 
[MinkUNet](https://github.com/NVIDIA/MinkowskiEngine), [SpUNet](https://github.com/traveller59/spconv) (see [SparseUNet](#sparseunet)),
[PTv1](https://arxiv.org/abs/2012.09164), [PTv2](https://arxiv.org/abs/2210.05666), [PTv3](https://arxiv.org/abs/2312.10035) (see [Point Transformers](#point-transformers)),
[Swin3D](https://github.com/microsoft/Swin3D) (see [Swin3D](#swin3d));   
**Instance Segmentation**: 
[PointGroup](https://github.com/dvlab-research/PointGroup) (see [PointGroup](#pointgroup)),  
[Panda Detector](https://arxiv.org/abs/2512.01324) (see [Panda Detector](#detector));  
**Pre-training**: 
[Sonata](https://arxiv.org/abs/2503.16429) (see [Sonata](#sonata));  
**Datasets**:
[PILArNet-M](https://arxiv.org/abs/2502.02558) (see [PILArNet-M](#pilarnet-m)) 

### TODO

We are looking at including the following models/modalities in the future:
- [ ] [SPINE](https://arxiv.org/abs/2102.01033), up until postprocessing module
- [ ] [PoLAr-MAE](https://arxiv.org/abs/2502.02558) pre-training and fine-tuning
- [ ] 2D TPC waveforms/networks, e.g., [NuGraph](https://arxiv.org/abs/2403.11872)
- [ ] Optical waveforms

## Installation

### Requirements
- Ubuntu: 18.04 and above
- CUDA: 11.3 and above (11.6+ recommended for FlashAttention support)
- PyTorch: 2.0.0 and above

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml --verbose
conda activate pimm-torch2.5.0-cu12.4
```

**FlashAttention** Requires CUDA 11.6+. If you cannot upgrade, disable FlashAttention in model configs by setting `enable_flash=False`.

## Directory Structure

Key directories in this repository:

- `configs/` - Configuration files for models, datasets, and training
- `pimm/` - Main codebase (models, datasets, training engine, utilities)
- `scripts/` - Training and testing shell scripts
- `tools/` - Python entry point scripts (`train.py`, `test.py`)
- `exp/` - Experiment outputs (logs, checkpoints, configs)
- `libs/` - External library dependencies

## Data Preparation

### 1. PILArNet Dataset

PILArNet has two revisions:

- **v1**: Original dataset from the PoLAr-MAE paper.

- **v2**: Reprocessed dataset with PID information, momentum, and vertex information used in the Panda paper.

To download either or both, do the following:

```python
# Download only v1; saved to ~/.cache/pimm/pilarnet/v1
python tools/download_pilarnet.py --version v1

# Download only v2; saved to ~/.cache/pimm/pilarnet/v2
python tools/download_pilarnet.py --version v2

# Save both to custom output directory
python tools/download_pilarnet.py --version both --output-dir /path/to/data
```

The events in v1 and v2 splits are different, so models trained on v1 should be evaluated on v1. All future models should be trained on v2.

### Environment Variables

Set the following environment variables to point to your PILArNet data:

```bash
export PILARNET_DATA_ROOT_V1="/path/to/pilarnet/v1/data"
export PILARNET_DATA_ROOT_V2="/path/to/pilarnet/v2/data"
```

Alternatively, create a `.env` file in the repository root:

```bash
PILARNET_DATA_ROOT_V1=/path/to/pilarnet/v1/data
PILARNET_DATA_ROOT_V2=/path/to/pilarnet/v2/data
```

The training scripts automatically source this file if it exists.

## Quick Start

### Single-GPU Training Example

For users with a single GPU, start with a simple training run:

```bash
# Single GPU training (fine-tuning for semantic segmentation)
sh scripts/train.sh -m 1 -g 1 -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin -n my_first_experiment
```

This will:
- Use 1 machine (`-m 1`) with 1 GPU (`-g 1`)
- Load config from `configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-lin.py`
- Save experiment outputs, including model checkpoints, to `exp/panda/semseg/my_first_experiment/`

If you want to save model checkpoints to a different directory that is more amenable to storing many large files, set the environment variable `MODEL_DIR=/path/to/model/dir/`. Model weights will be stored there, with a symbolic link to the experiment folder.

### Multi-GPU Training Example

For multi-GPU setups:

```bash
# Pre-training with 4 GPUs on 1 machine
sh scripts/train.sh -m 1 -g 4 -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask -n my_pretrain_exp

# Fine-tuning with pre-trained weights
sh scripts/train.sh -m 1 -g 4 -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin -n my_finetune_exp -w /path/to/checkpoint.pth
```

## Data Format

Point cloud data should be organized with the following structure:

```python
{
    'coord': (N, 3),           # 3D hit positions [x, y, z]
    'feat': (N, C),            # Hit features (charge, time, etc.)
    'segment': (N,1),          # Semantic labels (optional, for training)
    'instance': (N,1),         # Instance IDs (optional, for training)
}
```

The data often needs to be re-scaled to new domains that lead to more efficient training
(e.g., centering/scaling of coordinates to [-1,1]$^3$). This can be done within the Dataset class, or from a Transform. See the transform sections of configuration files for more details.

### Packed Data Format

This library works with packed data, where all batched quantities are in two dimensions instead of three, i.e. `(N, 3)` instead of `(B, N, 3)`. This is because point clouds are variable length, and getting to a 3 dimensional tensor would require padding. Instead of padding, there is an `offset` tensor, which is of length `B` and gives the indices in the packed tensors at which a point cloud ends and a new one starts.

`Offset` is conceptually similar to the concept of `Batch` in PyG, and can be seen as the cumulative sum of a `lengths` tensor. A visual illustration of batch and offset is as follows:

<p align="center">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png">
    <img alt="pointcept" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png" width="480">
    </picture><br>
</p>

## Configuration System

Configurations are Python dictionary-based files located in the `configs/` directory. Each config file defines the model architecture, dataset settings, training hyperparameters, and different hooks to run during training (checkpoint saving, logging, evaluation).

### Config Structure

Configs use a hierarchical structure with `_base_` inheritance:

```python
_base_ = ["../../_base_/default_runtime.py"]

# Override or add settings
model = dict(type="PT-v3m2", ...)
data = dict(train=dict(...), val=dict(...))
```

### Modifying Configs

You can modify configs in two ways:

1. Edit the config file directly
2. Override via command line using `--options`:
   ```bash
   sh scripts/train.sh ... -- --options epoch=50 data.train.max_len=500000
   ```

Example configs can be found in:
- `configs/panda/pretrain/` - Pre-training configurations
- `configs/panda/semseg/` - Semantic segmentation configurations  
- `configs/panda/panseg/` - Panoptic segmentation configurations

## Model Zoo

### SparseUNet

This repository provides `SparseUNet` implemented by `SpConv` and `MinkowskiEngine`. The SpConv version is recommended since SpConv is easy to install and faster than MinkowskiEngine. Meanwhile, SpConv is also widely applied in outdoor perception.

To use:
1. Install wither MinkowskiEngine or spconv (recommended)
3. Change the backbone in any config to `SpUNet-v1m1` or e.g. `MinkUNet50`. See [mink_unet.py](pimm/models/sparse_unet/mink_unet.py) for more model definitions.

### Point Transformers

- **PTv3**

[PTv3](https://arxiv.org/abs/2312.10035) is an efficient backbone model that achieves SOTA performances across indoor and outdoor scenarios. The full PTv3 relies on FlashAttention, while FlashAttention relies on CUDA 11.6 and above, make sure your local Pointcept environment satisfies the requirements. PTv3 also requires `spconv`.

If you can not upgrade your local environment to satisfy the requirements (CUDA >= 11.6), then you can disable FlashAttention by setting the model parameter `enable_flash` to `false`.

- **PTv2 mode2**

`PTv2 Mode2` enables AMP and disables _Position Encoding Multiplier_ & _Grouped Linear_. 

### Swin3D

[Swin3D](https://github.com/microsoft/Swin3D) is a hierarchical 3D Swin Transformer backbone.

To use:
1. Additional requirements:
```bash
# 1. Install MinkEngine v0.5.4, follow readme in https://github.com/NVIDIA/MinkowskiEngine;
# 2. Install Swin3D, mainly for cuda operation:
cd libs
git clone https://github.com/microsoft/Swin3D.git
cd Swin3D
pip install ./
```
2. Uncomment `# from .swin3d import *` in `pointcept/models/__init__.py`.
3. Change the backbone in any config to `Swin3D-v1m1`

### PointGroup

[PointGroup](https://github.com/dvlab-research/PointGroup) is an instance segmentation method that clusters points into object instances.

### Panda Detector

[Panda Detector](https://arxiv.org/abs/2512.01324) is a Mask2former-like objection detection framework for particle trajectories in LArTPC images.

### Sonata

[Sonata](https://arxiv.org/abs/2503.16429) is a discriminative self-supervised pre-training method similar to DINO for point clouds.

### Model Versioning

Models are versioned as e.g., `v1m2`, which corresponds to version 1 mode 2. We have:
- SparseUResNet, Point Transformer (V1-4), and Swin3D backbones
- Sonata-based pre-training and two object detection methods (PointGroup and Detector)

## Training

The entry point is `scripts/train.sh`. The script accepts the following key arguments:

- `-m`: Number of machines (nodes)
- `-g`: Number of GPUs per machine
- `-d`: Dataset/config directory (e.g., `panda/pretrain`, `panda/semseg`)
- `-c`: Config name (without `.py` extension)
- `-n`: Experiment name (used for output directory)
- `-w`: Path to checkpoint file (for fine-tuning/resuming)

### Pre-training Example

```bash
# Pre-training with Sonata on 1 machine with 4 GPUs
# Replace 'my_pretrain_exp' with your desired experiment name
sh scripts/train.sh -m 1 -g 4 -d panda/pretrain -c pretrain-sonata-v1m1-pilarnet-smallmask -n my_pretrain_exp
```

### Fine-tuning Examples

```bash
# Semantic segmentation using linear probing on a pre-trained weight
# Replace 'my_semseg_exp' with your experiment name and '/path/to/checkpoint.pth' with actual checkpoint path
sh scripts/train.sh -m 1 -g 4 -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin -n my_semseg_exp -w /path/to/checkpoint.pth

# Particle object detection using frozen encoder outputs
# 4 GPUs each on 2 machines (8 GPUs total)
sh scripts/train.sh -m 2 -g 4 -d panda/panseg -c detector-v1m1-pt-v3m2-ft-pid-dec -n my_detector_exp -w /path/to/checkpoint.pth

# Interaction-level object detection using frozen encoder outputs
sh scripts/train.sh -m 2 -g 4 -d panda/panseg -c detector-v1m1-pt-v3m2-ft-vtx-dec -n my_vtx_detector_exp -w /path/to/checkpoint.pth
```

### SLURM Cluster Training

For users on HPC clusters, SLURM scripts are found in `scripts/slurm/`. Example:

```bash
sbatch scripts/slurm/panseg/pilarnet_1node_amp_seed0_pid_dec_v1m1.sh
```

### Dataset Version Notes

- The PoLAr-MAE model was pre-trained and fine-tuned on v1
- The Panda model was pre-trained on v1, fine-tuned for semantic segmentation on v1, and fine-tuned for object detection on v2

## Inference/Testing

After training a model, you can evaluate it on test/validation sets using `scripts/test.sh`.

### Basic Usage

```bash
# Test on validation set
# -d: Dataset/config directory (must match training config)
# -c: Config name (must match training config)
# -n: Experiment name (must match training experiment name)
# -w: Weight file name (without .pth extension, e.g., 'model_best' or 'model_last')
sh scripts/test.sh -d panda/semseg -c semseg-pt-v3m2-pilarnet-ft-5cls-lin -n my_semseg_exp -w model_best
```

### Test Mode vs Train Mode

The test script runs the model in evaluation mode with:
- No data augmentation (deterministic transforms)
- Batch normalization in eval mode
- Gradient computation disabled
- Metrics computed on the full test/validation set

Test configurations are typically defined in the config file's `data.test` section, which may include different transforms optimized for inference (e.g., test-time augmentation, different voxelization strategies).

## Logging

Logging is available through either _Tensorboard_ or _Weights and Biases_ (recommended).
By default, both `tensorboard` and `wandb` are enabled. There are some usage notes related to `wandb`:
1. Disable by setting `use_wandb=False`;
2. Sync with  `wandb` remote server by `wandb login` in the terminal.
3. Set `wandb_project` in the config to set the wandb project to use.
4. Either set `wandb_run_name` or use `WandbNamer` to set the individual run name. `WandbNamer` is a hook which takes a set of defined config variables and sets them as the run name. E.g.,

```python
hooks = [
    dict(
        type="WandbNamer",
        keys=("model.type", "data.train.max_len", "amp_dtype", "seed"),
    ),
    ...
]
```

## Troubleshooting

### CUDA Version Compatibility

**Issue**: Errors related to CUDA version mismatch or FlashAttention not working.

**Solutions**:
- Ensure your CUDA version matches your PyTorch installation: `python -c "import torch; print(torch.version.cuda)"`
- For FlashAttention support, CUDA 11.6+ is required. If you cannot upgrade:
  - Set `enable_flash=False` in model configs

### Environment Variable Errors

**Issue**: `PILARNET_DATA_ROOT_V1/V2 is not set` error.

**Solutions**:
- Set environment variables: `export PILARNET_DATA_ROOT_V1=/path/to/data` 
- Ensure the paths point to directories containing `train/`, `val/`, and `test/` subdirectories with H5 files

### Importing issues

**Issue**: Can't import `pimm`. 

**Solutions**:
- **SpConv** (recommended): Ensure CUDA toolkit version matches PyTorch. Try: `pip install spconv-cu118` (adjust CUDA version)
- **MinkowskiEngine**: More complex to install. Consider using SpConv instead, which is easier and faster
- Verify CUDA is properly installed: `nvcc --version`

### Out of Memory (OOM) Errors

**Issue**: GPU runs out of memory during training.

**Solutions**:
- Reduce batch size in config
- Increase the number of GPUs used.
- Use mixed precision training (already enabled by default with `enable_amp=True`)

### Checkpoint Loading Errors

**Issue**: Cannot load checkpoint or checkpoint path not found.

**Solutions**:
- Use absolute paths for checkpoint files: `-w /full/path/to/checkpoint.pth`
- Check that the checkpoint file exists and is not corrupted
- Ensure the checkpoint matches the model architecture in your config
- For resuming training, use `-r true` flag: `sh scripts/train.sh ... -r true`

## Acknowledgements

This codebase is built on [Pointcept](https://github.com/Pointcept/Pointcept) and adapted for TPC data. We thank the Pointcept team for their excellent framework.

## License

This project inherits the MIT license from Pointcept.
