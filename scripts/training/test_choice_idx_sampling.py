#!/usr/bin/env python3
"""验证 stage1 DataLoader 与 train_helios.py 一致的 choice_idx 抽样。

对比场景（尽量复现 stage_1_init.yaml + train_helios 建表逻辑）：
  1. shared_epoch + yaml 的 num_workers / persistent_workers（正式训练路径）
  2. 无 shared_epoch + 同上 worker 配置（persistent worker 下 epoch 项恒为 0 的复现）
  3. num_workers=0（主进程读 _epoch，作对照）

用法（在 Helios 仓库根目录）:
  python scripts/training/test_choice_idx_sampling.py \\
      --config scripts/training/configs/stage_1_init.yaml \\
      --num-epochs 2
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

# 仓库根目录 = Helios/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helios.dataset.dataloader_history_latents_dist import (  # noqa: E402
    BucketedFeatureDataset,
    BucketedSampler,
    collate_fn,
)
from helios.utils.train_config import Args  # noqa: E402


def _load_conf(config_path: str):
    schema = OmegaConf.structured(Args)
    config = OmegaConf.load(config_path)
    return OmegaConf.merge(schema, config)


def _dataset_sampling_ratios(conf) -> dict | None:
    ratios = conf.data_config.dataset_sampling_ratios
    if not ratios:
        return None
    return {item["dataset_name"].rstrip("/"): item["ratio"] for item in ratios}


def build_train_like_dataloader(
    conf,
    *,
    use_shared_epoch: bool,
    num_workers: int | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
):
    """与 train_helios.py stage1 分支相同的 DataLoader 构造。"""
    nw = conf.data_config.dataloader_num_workers if num_workers is None else num_workers
    pw = conf.data_config.persistent_workers if persistent_workers is None else persistent_workers
    pf = conf.data_config.prefetch_factor if prefetch_factor is None else prefetch_factor

    shared_epoch = multiprocessing.Value("i", 0) if use_shared_epoch else None
    dataset_kwargs = {
        "feature_folders": conf.data_config.instance_data_root,
        "single_res": conf.data_config.single_res,
        "single_height": conf.data_config.single_height,
        "single_width": conf.data_config.single_width,
        "return_prompt_raw": conf.training_config.is_use_reward_model,
        "return_all_vae_latent": (
            conf.training_config.dmd_teacher_forcing and conf.training_config.dmd_teacher_forcing_ratio > 0
        )
        or conf.training_config.is_use_gan,
        "history_sizes": conf.training_config.history_sizes,
        "is_keep_x0": True,
        "force_rebuild": conf.data_config.force_rebuild,
        "seed": conf.seed,
        "shared_epoch": shared_epoch,
    }
    dataset = BucketedFeatureDataset(**dataset_kwargs)
    sampler = BucketedSampler(
        dataset,
        batch_size=conf.training_config.train_batch_size,
        drop_last=True,
        shuffle=conf.data_config.use_shuffle,
        seed=conf.seed,
        dataset_sampling_ratios=_dataset_sampling_ratios(conf),
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        pin_memory=conf.data_config.pin_memory,
        prefetch_factor=pf if nw > 0 and pf > 0 else None,
        persistent_workers=pw if nw > 0 else False,
        collate_fn=collate_fn,
        num_workers=nw,
    )
    return dataset, sampler, loader, shared_epoch


def collect_records(dataset, sampler, loader, num_epochs: int) -> list[dict]:
    records = []
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        for step, batch in enumerate(loader):
            choice = batch["choice_idx"]
            if isinstance(choice, torch.Tensor):
                choice = choice.tolist()
            records.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "choice_idx": list(choice),
                    "uttid": list(batch["uttid"]),
                }
            )
    return records


def _choice_key(rec: dict) -> tuple:
    return tuple(rec["choice_idx"])


def summarize(name: str, records: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(f"场景: {name}")
    print(f"{'=' * 72}")
    if not records:
        print("  (无 batch)")
        return

    for rec in records[: min(12, len(records))]:
        print(
            f"  epoch={rec['epoch']:3d} step={rec['step']:4d} "
            f"choice_idx={rec['choice_idx']} uttid={rec['uttid'][0]!r}"
        )
    if len(records) > 12:
        print(f"  ... 共 {len(records)} 个 batch，仅打印前 12 条")

    # 单样本：同一 uttid 跨 epoch 的 choice_idx 是否变化
    by_uttid_epoch = defaultdict(dict)
    for rec in records:
        uttid = rec["uttid"][0]
        by_uttid_epoch[uttid][rec["epoch"]] = _choice_key(rec)

    print("\n  按 uttid × epoch 汇总 choice_idx:")
    for uttid, epoch_map in by_uttid_epoch.items():
        parts = [f"e{e}={epoch_map[e]}" for e in sorted(epoch_map)]
        print(f"    {uttid}: {', '.join(parts)}")
        if len(epoch_map) >= 2:
            vals = list(epoch_map.values())
            if vals[0] == vals[1]:
                print("      → 前两个 epoch 的 choice_idx 相同（无 shared_epoch + persistent_workers 时常见）")
            else:
                print("      → 前两个 epoch 的 choice_idx 不同（shared_epoch 或 num_workers=0 时预期）")


def main():
    parser = argparse.ArgumentParser(description="验证 choice_idx 与 train_helios 一致的抽样行为")
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/training/configs/stage_1_init.yaml",
        help="与 train_helios.py 相同的 OmegaConf 配置",
    )
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["shared", "no_shared", "workers0"],
        choices=["shared", "no_shared", "workers0", "all"],
        help="shared=训练默认; no_shared=去掉 shared_epoch; workers0=主进程对照",
    )
    args = parser.parse_args()
    if "all" in args.scenarios:
        args.scenarios = ["shared", "no_shared", "workers0"]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    conf = _load_conf(str(config_path))

    print("配置摘要（与 train_helios stage1 对齐）:")
    print(f"  seed={conf.seed}")
    print(f"  instance_data_root={conf.data_config.instance_data_root}")
    print(f"  train_batch_size={conf.training_config.train_batch_size}")
    print(f"  num_workers={conf.data_config.dataloader_num_workers}")
    print(f"  persistent_workers={conf.data_config.persistent_workers}")
    print(f"  prefetch_factor={conf.data_config.prefetch_factor}")
    print(f"  use_shuffle={conf.data_config.use_shuffle}")
    print(f"  history_sizes={conf.training_config.history_sizes}")

    scenarios = []
    if "shared" in args.scenarios:
        scenarios.append(
            (
                "A) shared_epoch + yaml workers（= train_helios.py）",
                dict(use_shared_epoch=True),
            )
        )
    if "no_shared" in args.scenarios:
        scenarios.append(
            (
                "B) 无 shared_epoch + yaml workers（epoch 项在 worker 内可能恒为 0）",
                dict(use_shared_epoch=False),
            )
        )
    if "workers0" in args.scenarios:
        scenarios.append(
            (
                "C) shared_epoch + num_workers=0（主进程对照）",
                dict(use_shared_epoch=True, num_workers=0, persistent_workers=False),
            )
        )

    for title, kwargs in scenarios:
        dataset, sampler, loader, _ = build_train_like_dataloader(conf, **kwargs)
        print(f"\n构建 DataLoader: len(dataset)={len(dataset)}, batches/epoch≈{len(loader)}")
        records = collect_records(dataset, sampler, loader, args.num_epochs)
        summarize(title, records)

    print(
        "\n说明: choice_idx 由 prepare_stage1_latent 中 "
        "sample_seed = base_seed + epoch*1e6 + idx 决定；"
        "idx 为 samples 列表下标（单条数据时恒为 0）。"
    )


if __name__ == "__main__":
    main()
