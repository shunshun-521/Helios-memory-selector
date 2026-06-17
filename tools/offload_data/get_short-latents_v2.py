"""
T2A (Text-to-Audio) data preprocessing script.
Encodes audio files into latents using MOVA DAC VAE and text embeddings using UMT5.

Usage:
    torchrun --nproc_per_node=1 tools/offload_data/get_short-latents_v2.py \
        --pretrained_model_name_or_path /root/autodl-tmp/mova-weight \
        --audio_vae_path /root/autodl-tmp/mova-weight

Audio JSON format (each entry):
    {
        "id": "unique_id",
        "audio_path": "relative/path/to/audio.wav",
        "prompt": "text description of the audio",
        "duration": 10.0,
        "sample_rate": 48000
    }
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
import torchaudio
from accelerate import Accelerator
from helios.dataset.dac_vae import DAC
from helios.dataset.dataloader_wav_dist import BucketedFeatureDataset, BucketedSampler, collate_fn
from helios.utils.utils_base import encode_prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.training_utils import free_memory


# ======================== Main ========================

def main(
    rank,
    world_size,
    global_rank,
    batch_size,
    dataloader_num_workers,
    json_file,
    audio_folder,
    output_latent_folder,
    pretrained_model_name_or_path,
    audio_vae_path,
    target_sr,
    max_duration,
    latent_window_size,
):
    weight_dtype = torch.bfloat16
    device = rank
    seed = 42

    # Load text encoder
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype, local_files_only=True
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(device)

    # Load audio VAE (DAC)
    audio_vae = DAC.from_pretrained(audio_vae_path).to(device=device)
    audio_vae.eval()
    audio_vae.requires_grad_(False)

    # Dataset & Dataloader
    dataset = BucketedFeatureDataset(
        json_files=[json_file],
        audio_folders=[audio_folder],
        target_sample_rate=target_sr,
        target_duration=max_duration,
        force_rebuild=False,
    )
    sampler = BucketedSampler(
        dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        shuffle=True, 
        seed=seed,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=global_rank,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )

    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")

    os.makedirs(output_latent_folder, exist_ok=True)

    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Processing audio")

    for idx, batch in enumerate(dataloader):
        if batch is None:
            if rank == 0:
                pbar.update(1)
            continue

        free_memory()

        # Skip already processed
        valid_mask = []
        for i, uttid in enumerate(batch["uttid"]):
            duration = batch["audio_metadata"]["duration"][i]
            # Calculate seq_len based on duration, target_sr and DAC hop_length (960)
            seq_len = int(duration * target_sr / 960)
            output_path = os.path.join(output_latent_folder, f"{uttid}_{seq_len}.pt")
            valid_mask.append(not os.path.exists(output_path))

        if not any(valid_mask):
            if rank == 0:
                pbar.update(1)
            continue

        with torch.no_grad():
            # Encode audio -> latent
            # waveform: (B, 1, T_samples)
            waveform = batch["audios"].to(device=device, dtype=torch.float32)
            
            # Use MOVA DAC VAE:
            # First encode to get the continuous distribution, then take the mode (mean)
            posterior, _, _, _, _ = audio_vae.encode(waveform)
            audio_latent = posterior.mode()  # (B, C_latent, T_latent)

            # Split latent into chunks of latent_window_size along time dim
            T_latent = audio_latent.shape[2]
            num_chunks = T_latent // latent_window_size
            if num_chunks == 0:
                num_chunks = 1

            chunked = audio_latent[:, :, :num_chunks * latent_window_size]
            # (B, C, num_chunks * ws) -> (B, num_chunks, C, ws)
            vae_latents = chunked.unfold(2, latent_window_size, latent_window_size).permute(0, 2, 1, 3)

            # Encode text prompts
            prompt_embeds, prompt_attention_mask = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=batch["prompts"],
                device=device,
            )

        # Save per-sample
        for i, uttid in enumerate(batch["uttid"]):
            if not valid_mask[i]:
                continue

            duration = batch["audio_metadata"]["duration"][i]
            seq_len = int(duration * target_sr / 960)
            output_path = os.path.join(output_latent_folder, f"{uttid}_{seq_len}.pt")

            temp_to_save = {
                "vae_latent": vae_latents[i].cpu().detach(),       # (num_chunks, C_latent, T_chunk)
                "prompt_embed": prompt_embeds[i].cpu().detach(),    # (seq_len, hidden_dim)
                "prompt_raw": batch["prompts"][i],
                "num_samples": batch["audio_metadata"]["num_samples"][i],
                "sample_rate": target_sr,
            }
            try:
                torch.save(temp_to_save, output_path)
                print(f"Saved: {output_path} | latent shape: {vae_latents[i].shape}")
            except Exception as e:
                print(f"Error saving {output_path}: {e}")

        if rank == 0:
            pbar.update(1)

        # Cleanup
        del waveform, audio_latent, vae_latents, prompt_embeds, batch
        free_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2A audio preprocessing: encode audio to latents + text embeddings")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/root/autodl-tmp/mova-weight",
                        help="Path to Helios model (for text encoder)")
    parser.add_argument("--audio_vae_path", type=str, default="/root/autodl-tmp/mova-weight",
                        help="Path to MOVA DAC VAE model")
    parser.add_argument("--json_file", type=str, required=True,default="/root/autodl-tmp/Helios/data/mova_audio_infer/t2a_output/new_train.json",
                        help="JSON file with audio metadata (id, audio_path, prompt, duration)")
    parser.add_argument("--audio_folder", type=str, required=True,default="/root/autodl-tmp/Helios/data/mova_audio_infer/t2a_output/wav",
                        help="Base folder containing audio files")
    parser.add_argument("--output_latent_folder", type=str, required=True,default="/root/autodl-tmp/Helios/data/mova_audio_infer/latent data",
                        help="Output folder for .pt latent files")
    parser.add_argument("--target_sr", type=int, default=48000, help="Target sample rate")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Max audio duration in seconds")
    parser.add_argument("--latent_window_size", type=int, default=9, help="Latent chunk size along time dim")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    device = torch.cuda.current_device()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    main(
        rank=device,
        world_size=world_size,
        global_rank=global_rank,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        json_file=args.json_file,
        audio_folder=args.audio_folder,
        output_latent_folder=args.output_latent_folder,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        audio_vae_path=args.audio_vae_path,
        target_sr=args.target_sr,
        max_duration=args.max_duration,
        latent_window_size=args.latent_window_size,
    )

    dist.barrier()
    dist.destroy_process_group()
