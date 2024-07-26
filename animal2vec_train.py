#!/usr/bin/env python3 -u
# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import hydra
import torch
import logging

from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from fairseq import utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.distributed import utils as distributed_utils
from fairseq.logging import metrics
from nn import animal2vec_audio_main

logger = logging.getLogger("animal2vec.hydra_train")


@hydra.main(config_path=os.path.join(os.path.curdir, "configs"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    _hydra_main(cfg)


def _hydra_main(cfg: FairseqConfig, **kwargs) -> float:
    # print(cfg)
    add_defaults(cfg)

    if cfg.common.reset_logging:
        utils.reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, animal2vec_audio_main, **kwargs)
        else:
            distributed_utils.call_main(cfg, animal2vec_audio_main, **kwargs)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))

    # get best val and return - useful for sweepers
    try:
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        best_val = None

    if best_val is None:
        best_val = float("inf")

    return best_val


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
