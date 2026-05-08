"""
Audio (WAV) dataloader for T2A, adapted from dataloader_mp4_dist.py.
Used by get_audio-latents.py to load raw audio + caption for VAE encoding.

Key differences from video version:
- Reads .wav instead of .mp4
- bucket_key = (num_samples, sample_rate) instead of (num_frame, height, width)
- No spatial crop/resize, only temporal cut + resample
- Returns waveform tensor (1, T_samples) instead of video tensor (T, C, H, W)
- duration trained on is 193/24s (≈8.04s) to match video frames, but can be adjusted with --target_duration
"""

import json
import os
import pickle
import random
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


def find_nearest_duration_bucket(duration, target_duration=193/24.0):
    """Round duration down to nearest 0.5s bucket."""
    return min(duration, target_duration)


class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        json_files,
        audio_folders,
        target_sample_rate=48000,
        target_duration=10.0,
        force_rebuild=True,
        single_res=False,
        single_sample_rate=48000,
        single_duration=193/24,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.force_rebuild = force_rebuild
        self.single_res = single_res
        self.single_sample_rate = single_sample_rate
        self.single_duration = single_duration
        self._epoch = 0

        if isinstance(json_files, str):
            self.json_files = [json_files]
        else:
            self.json_files = json_files

        if isinstance(audio_folders, str):
            self.audio_folders = [audio_folders]
        else:
            self.audio_folders = audio_folders

        assert len(self.json_files) == len(self.audio_folders)

        self.samples = []
        self.buckets = defaultdict(list)

        for json_file, audio_folder in zip(self.json_files, self.audio_folders):
            cache_file = json_file.replace(".json", "_wav_cache.pkl")
            self._process_json_file(json_file, audio_folder, cache_file)

    def _process_json_file(self, json_file, audio_folder, cache_file):
        if self.force_rebuild or not os.path.exists(cache_file):
            if os.path.exists(cache_file):
                os.remove(cache_file)
            print(f"Building metadata cache for: {json_file}")
            file_samples, file_buckets = self._build_file_metadata(json_file, audio_folder)
            cached_data = {"samples": file_samples, "buckets": file_buckets}
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"Cached {len(file_samples)} samples from {json_file}\n")
        else:
            print(f"Loading cached metadata from: {cache_file}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            file_samples = cached_data["samples"]
            file_buckets = cached_data["buckets"]
            print(f"Loaded {len(file_samples)} samples from cache\n")

        sample_idx_offset = len(self.samples)
        self.samples.extend(file_samples)
        for bucket_key, indices in file_buckets.items():
            self.buckets[bucket_key].extend([idx + sample_idx_offset for idx in indices])

    def _build_file_metadata(self, json_file, audio_folder):
        with open(json_file, "r") as f:
            data = json.load(f)

        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        print(f"Processing {len(data)} records from {json_file}...")
        for i, item in enumerate(data):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(data)} records")

            wav_path = item["path"]
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(audio_folder, wav_path)

            if not os.path.exists(wav_path):
                continue

            duration = item.get("duration", self.target_duration)
            sample_rate = item.get("sample_rate", self.target_sample_rate)
            cut = item.get("cut", [0.0, duration])
            cap = item.get("cap", [""])
            prompt = cap[0] if isinstance(cap, list) else cap

            cut_duration = cut[1] - cut[0]
            if cut_duration < 1.0:
                continue

            uttid = os.path.basename(wav_path).replace(".wav", "")

            # For T2A, bucket by (target_num_samples, sample_rate)
            # Simplify: all same duration/sr → single bucket
            target_num_samples = int(self.target_duration * self.target_sample_rate)
            bucket_key = (target_num_samples, self.target_sample_rate)

            sample_info = {
                "uttid": uttid,
                "dataset_name": json_file.rstrip("/"),
                "audio_folder": audio_folder,
                "audio_path": wav_path,
                "bucket_key": bucket_key,
                "prompt": prompt,
                "sample_rate": sample_rate,
                "duration": duration,
                "cut_start": cut[0],
                "cut_end": cut[1],
                "target_num_samples": target_num_samples,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_retries = 100
        for _ in range(max_retries):
            sample_info = self.samples[idx]
            try:
                waveform, sr = torchaudio.load(sample_info["audio_path"])

                # Temporal cut (in seconds → samples)
                start_sample = int(sample_info["cut_start"] * sr)
                end_sample = int(sample_info["cut_end"] * sr)
                waveform = waveform[:, start_sample:end_sample]

                # Resample if needed
                if sr != self.target_sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)

                # Mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Pad or truncate to target length
                target_len = sample_info["target_num_samples"]
                if waveform.shape[1] < target_len:
                    waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
                else:
                    waveform = waveform[:, :target_len]

                return {
                    "uttid": sample_info["uttid"],
                    "bucket_key": sample_info["bucket_key"],
                    "dataset_name": sample_info["dataset_name"],
                    "audio_metadata": {
                        "num_samples": sample_info["target_num_samples"],
                        "sample_rate": self.target_sample_rate,
                        "duration": sample_info["duration"],
                    },
                    "audios": waveform,  # (1, T_samples)
                    "prompts": sample_info["prompt"],
                }
            except Exception as e:
                print(f"Error loading {sample_info['audio_path']}: {e}")
                idx = random.randint(0, len(self.samples) - 1)

        print(f"Failed after {max_retries} retries")
        return None


class BucketedSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        shuffle=True,
        seed=42,
        dataset_sampling_ratios=None,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator()
        self.buckets = dataset.buckets
        self._epoch = 0

        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank
        self.ith_sp_group = self.global_rank // self.sp_world_size

        self.dataset_sampling_ratios = (
            {key.rstrip("/"): value for key, value in dataset_sampling_ratios.items()}
            if dataset_sampling_ratios is not None
            else {}
        )
        self._prepare_dataset_buckets()

    def _prepare_dataset_buckets(self):
        self.dataset_buckets = {}
        for bucket_key, sample_indices in self.buckets.items():
            dataset_groups = {}
            for idx in sample_indices:
                dataset_name = self.dataset.samples[idx]["dataset_name"]
                if dataset_name not in dataset_groups:
                    dataset_groups[dataset_name] = []
                dataset_groups[dataset_name].append(idx)
            self.dataset_buckets[bucket_key] = dataset_groups

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _shard_indices_for_sp_group(self, indices):
        if self.num_sp_groups == 1:
            return indices
        indices_tensor = torch.tensor(indices, dtype=torch.long) if isinstance(indices, list) else indices
        total_size = len(indices_tensor)
        if total_size % self.num_sp_groups != 0:
            if not self.drop_last:
                padding_size = self.num_sp_groups - (total_size % self.num_sp_groups)
                indices_tensor = torch.cat([indices_tensor, indices_tensor[:padding_size]])
            elif self.drop_last:
                indices_tensor = indices_tensor[: (total_size // self.num_sp_groups) * self.num_sp_groups]
        return indices_tensor[self.ith_sp_group :: self.num_sp_groups].tolist()

    def __iter__(self):
        epoch_seed = self.seed + self._epoch
        self.generator.manual_seed(epoch_seed)

        bucket_iterators = {}
        for bucket_key, dataset_groups in self.dataset_buckets.items():
            balanced_indices = sum(dataset_groups.values(), [])
            if self.shuffle:
                perm = torch.randperm(len(balanced_indices), generator=self.generator).tolist()
                balanced_indices = [balanced_indices[i] for i in perm]
            sp_group_indices = self._shard_indices_for_sp_group(balanced_indices)

            batches = []
            for i in range(0, len(sp_group_indices), self.batch_size):
                batch = sp_group_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
            if batches:
                bucket_iterators[bucket_key] = iter(batches)

        remaining = list(bucket_iterators.keys())
        while remaining:
            idx = torch.randint(len(remaining), (1,), generator=self.generator).item()
            bucket_key = remaining[idx]
            try:
                yield next(bucket_iterators[bucket_key])
            except StopIteration:
                remaining.remove(bucket_key)

    def __len__(self):
        total_batches = 0
        for bucket_key, dataset_groups in self.dataset_buckets.items():
            n = sum(len(v) for v in dataset_groups.values())
            sp_n = n // self.num_sp_groups + (1 if not self.drop_last and n % self.num_sp_groups else 0)
            total_batches += sp_n // self.batch_size + (1 if not self.drop_last and sp_n % self.batch_size else 0)
        return total_batches


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    def _collate(data_list):
        if isinstance(data_list[0], dict):
            return {key: _collate([d[key] for d in data_list]) for key in data_list[0]}
        elif isinstance(data_list[0], torch.Tensor):
            return torch.stack(data_list)
        else:
            return data_list

    return {key: _collate([d[key] for d in batch]) for key in batch[0]}
