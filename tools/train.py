"""
Distributed Training Script (Native)

This script is designed to run as a native distributed process.
It works seamlessly with SLURM (where Slurm launches tasks) or
falls back to a single-process local run for debugging.

Author: Modified from original larfm train.py
"""

import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from larfm.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from larfm.engines.train import TRAINERS
from larfm.utils import comm

def main_worker(cfg):
    """Main worker function that runs the training."""
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    
    comm.setup_distributed()
    main_worker(cfg)

if __name__ == "__main__":
    main()
