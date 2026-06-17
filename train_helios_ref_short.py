#!/usr/bin/env python3
"""Ref-Short Stage1 post training entry (wraps train_helios.main)."""

import argparse
import os

from omegaconf import OmegaConf

from helios.utils.train_config import Args


def main():
    parser = argparse.ArgumentParser(description="Train Helios with ref-short path (patch_ref + SelectorRuntime)")
    parser.add_argument("--config", type=str, required=True, help="YAML config, e.g. scripts/training/configs/stage_1_ref_short.yaml")
    args_cli = parser.parse_args()

    config = OmegaConf.load(args_cli.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)

    conf.training_config.use_ref_short = True
    conf.data_config.return_vae_latent_for_ref = True
    conf.training_config.has_multi_term_memory_patch = True
    conf.training_config.zero_history_timestep = True

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != conf.training_config.local_rank:
        conf.training_config.local_rank = env_local_rank

    from train_helios import main as train_main

    train_main(conf)


if __name__ == "__main__":
    main()
