# LAr.FM - Liquid Argon Foundation Models

A codebase for representation learning and perception in Liquid Argon Time Projection Chambers (LArTPCs), built on the [Pointcept](https://github.com/Pointcept/Pointcept) framework.

## Overview

**LAr.FM** adapts 3D point cloud methods for neutrino event reconstruction in LArTPC detectors. This repository provides:

- **Self-supervised pre-training** (Sonata) for LArTPC hit clouds
- **Panoptic segmentation** models for particle instance and semantic segmentation
- **Semantic segmentation** for motif-level (track, shower, ...) per-pixel segmentation.

## Installation

### Requirements
- Ubuntu: 18.04 and above
- CUDA: 11.3 and above
- PyTorch: 2.0.0 and above

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml --verbose
conda activate pointcept-torch2.5.0-cu12.4
```
## Quick Start

TBD 

## Data Format

LArTPC hit data should be organized with the following structure:

```python
{
    'coord': (N, 3),          # 3D hit positions [x, y, z]
    'feat': (N, C),           # Hit features (charge, time, etc.)
    'segment': (N,),          # Semantic labels (optional, for training)
    'instance': (N,),         # Instance IDs (optional, for training)
    'momentum': (N,),         # True momentum (optional, for training)
}
```

## Distributed Training

```bash
# Single node, 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh scripts/train.sh -g 8 -d panda -c panseg/detector-v1m1-pt-v3m2-ft-pid-dec -n exp_name

# SLURM cluster
sbatch scripts/slurm/panseg/pilarnet_1node_amp_seed0_pid_dec_v1m1.sh
```

## Citation

This work builds on Pointcept. If you use this codebase, please cite:

```bibtex
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

For Sonata pre-training:
```bibtex
@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and others},
    booktitle={CVPR},
    year={2025}
}
```

For Point Transformer V3:
```bibtex
@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and others},
    booktitle={CVPR},
    year={2024}
}
```

## Acknowledgements

This codebase is built on [Pointcept](https://github.com/Pointcept/Pointcept) and adapted for LArTPC detector analysis. We thank the Pointcept team for their excellent framework.

## License

This project inherits the license from Pointcept. Please refer to the original repository for licensing details.
