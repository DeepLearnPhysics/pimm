"""
Configuration for pretraining a SONATA model on PILArNet dataset

This config file adapts the SONATA pretraining approach for particle physics point clouds.
"""

_base_ = ["./pretrain-sonata-v1m1-pilarnet-1m-amp-seed0-smallmask.py"]

num_events = 1_000
# scheduler settings
epoch = 10 * (1_000_000 // num_events)
wandb_run_name = f"sonata-large-pilarnet-pretrain-4GPU-{num_events}ev-amp-seed0-NormCoords-logE-{epoch}epoch"  # Descriptive name for this run
eval_epoch = 10 * (1_000_000 // num_events)
data = dict(
    train=dict(
        max_len=num_events,
    ),
)