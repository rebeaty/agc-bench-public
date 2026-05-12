"""Pre-merge AGC-Judge (LoRA on Qwen3-30B-A3B-Instruct-2507) into a flat
checkpoint, so HF Inference Endpoints / TGI / vLLM can serve it without
runtime LoRA loading. This avoids deployment incompatibilities between TGI
runtime LoRA loading and Qwen3-MoE attention naming.

Loads the base sharded across all visible GPUs via device_map="auto",
applies the LoRA via peft, calls merge_and_unload(), saves locally.
Optionally uploads to HF as `<namespace>/AGC-Judge-merged`.

Usage:
  python scripts/merge_agc_judge_lora.py
  python scripts/merge_agc_judge_lora.py --upload-to <hf-namespace>/AGC-Judge-merged --private

Memory note: Qwen3-30B-A3B in bf16 is about 60GB. Sharding across 3 x A6000
(3 x 48GB) gives roughly 20GB per card with comfortable headroom.

Output: `analysis/agc_judge_merged/` locally (~60GB on disk).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_LORA = "agcbench-2026/AGC-Judge"
DEFAULT_OUT = REPO / "analysis" / "agc_judge_merged"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--lora', default=DEFAULT_LORA)
    p.add_argument('--out', default=str(DEFAULT_OUT))
    p.add_argument('--max-memory-per-gpu', default='38GiB',
                   help='per-GPU memory cap for sharding (leave headroom for other users)')
    p.add_argument('--cpu-memory', default='200GiB',
                   help='CPU memory cap for offload')
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16'])
    p.add_argument('--upload-to', default=None,
                   help='if set (e.g. "<hf-namespace>/AGC-Judge-merged"), upload after merging')
    p.add_argument('--private', action='store_true', help='make uploaded repo private')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print('ERROR: no CUDA devices visible.', file=sys.stderr)
        return 2
    print(f'GPUs visible: {n_gpus}')
    for i in range(n_gpus):
        free, total = torch.cuda.mem_get_info(i)
        print(f'  cuda:{i}  free={free/1e9:.1f}GB  total={total/1e9:.1f}GB  ({torch.cuda.get_device_name(i)})')

    max_memory = {i: args.max_memory_per_gpu for i in range(n_gpus)}
    max_memory['cpu'] = args.cpu_memory
    print(f'\nmax_memory: {max_memory}')

    # Lazy imports so --help doesn't pay the load cost
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    print(f'\n[1/4] Loading base model {args.base!r} sharded across {n_gpus} GPU(s)…')
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map='auto',
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f'  done in {time.time()-t0:.1f}s')

    print(f'\n[2/4] Loading tokenizer + applying LoRA {args.lora!r}…')
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.lora, is_trainable=False)
    print(f'  done in {time.time()-t0:.1f}s')

    print(f'\n[3/4] Merging adapter into base (merge_and_unload)…')
    t0 = time.time()
    merged = model.merge_and_unload()
    print(f'  done in {time.time()-t0:.1f}s')

    print(f'\n[4/4] Saving merged checkpoint to {out!r}…')
    t0 = time.time()
    merged.save_pretrained(out, safe_serialization=True, max_shard_size='5GB')
    tokenizer.save_pretrained(out)
    print(f'  done in {time.time()-t0:.1f}s')
    print(f'\nMerged checkpoint at: {out}')

    if args.upload_to:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f'\n[5/5] Uploading to {args.upload_to} (private={args.private})…')
        api.create_repo(args.upload_to, repo_type='model', private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.upload_to, repo_type='model')
        print(f'  uploaded → https://huggingface.co/{args.upload_to}')


if __name__ == '__main__':
    main()
