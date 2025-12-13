"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
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
from larfm.engines.test import TESTERS
from larfm.utils import comm


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    logging.basicConfig(level=logging.INFO)
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    comm.setup_distributed()
    main_worker(cfg)


if __name__ == "__main__":
    main()
