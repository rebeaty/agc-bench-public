"""Spin up an HF Inference Endpoint for AGC-Judge.

The model `agcbench-2026/AGC-Judge` is a LoRA adapter on
`Qwen/Qwen3-30B-A3B-Instruct-2507` (30B-MoE, 3B active params).
This script encodes the deployment configuration used for the release
reproduction checks:

  - **Base**: Qwen/Qwen3-30B-A3B-Instruct-2507 (TGI loads it; the LoRA adapter
    is mounted via the LORA_ADAPTERS env var)
  - **Container**: TGI v3 (text-generation-inference) — exposes `/v1/chat/completions`
    natively, supports continuous batching, and accepts LORA_ADAPTERS for
    runtime-loaded adapters. The HF *default* text-generation handler does NOT
    expose `/v1/chat/completions`, so HELM's OpenAI-style client cannot reach it.
  - **GPU**: H200 in `aws-us-east-2` (141GB).
    This single-GPU profile has enough VRAM for the 16-bit base model plus
    tokenizer, KV cache, and LoRA adapter.
  - **Scale-to-zero**: 15 min idle → $0/hr while not serving.

Pre-flight requirements:

  1. **HF Inference Endpoints billing must be activated separately from any
     existing HF subscription.** A card-on-file on the Hub side is not enough.
     Go to https://endpoints.huggingface.co/ once and complete the billing
     onboarding flow (accept ToS + confirm card). Without this, every deploy
     attempt 403s with `Payment method required for namespace: <user>`.

  2. **Endpoint GPU quota is allocated by HF account.** If creation fails for
     quota or billing reasons, resolve that in the HF Endpoint console before
     re-running this script.

Cost model:
  HF Inference Endpoint pricing, instance availability, and quota are
  account-, vendor-, region-, and date-dependent. Review the endpoint console
  and https://huggingface.co/docs/inference-endpoints/pricing before creating
  the endpoint.

Idempotent: running this when an endpoint already exists (by name) reuses it
if healthy, otherwise tears it down and recreates. The endpoint namespace
defaults to your logged-in HF user.

Usage:
  python eval_scripts/spin_up_hf_endpoint.py
  python eval_scripts/spin_up_hf_endpoint.py --instance-type nvidia-h200 --region us-east-2
  python eval_scripts/spin_up_hf_endpoint.py --namespace your-org

After it returns, prod_env/model_deployments.yaml is updated to point at the
live endpoint URL, and you can run:
  ./eval_scripts/run_with_agc_judge.sh <model>
"""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, get_inference_endpoint, InferenceEndpointError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENTS_YAML = REPO_ROOT / 'prod_env' / 'model_deployments.yaml'

# Tunables — change defaults via CLI flags.
DEFAULT_NAME = "agc-judge"
# Namespace defaults to the logged-in HF user (BYO-key model: you pay HF
# directly for your own endpoint). Override with --namespace if you want to
# deploy into an org you belong to.
DEFAULT_NAMESPACE = None
DEFAULT_INSTANCE_TYPE = "nvidia-h200"   # 141GB; comfortable headroom for Qwen3-30B 16-bit
DEFAULT_INSTANCE_SIZE = "x1"
DEFAULT_REGION = "us-east-2"            # H200 is only here on HF/AWS
DEFAULT_VENDOR = "aws"
DEFAULT_SCALE_TO_ZERO_MIN = 15

# TGI v3 image with LoRA-adapter support (LORA_ADAPTERS env var). Used for
# the runtime; the LoRA repo is mounted via env, the base model is the repo
# we point HF at on create.
DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_LORA_REPO = "agcbench-2026/AGC-Judge"
DEFAULT_TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference:3.0.0"


def _confirm(prompt: str) -> bool:
    print(prompt, end=' ', flush=True)
    return sys.stdin.readline().strip().lower() in ('y', 'yes')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', default=DEFAULT_NAME, help='endpoint name (default: agc-judge)')
    p.add_argument('--namespace', default=DEFAULT_NAMESPACE)
    p.add_argument('--instance-type', default=DEFAULT_INSTANCE_TYPE)
    p.add_argument('--instance-size', default=DEFAULT_INSTANCE_SIZE)
    p.add_argument('--region', default=DEFAULT_REGION)
    p.add_argument('--vendor', default=DEFAULT_VENDOR)
    p.add_argument('--scale-to-zero-min', type=int, default=DEFAULT_SCALE_TO_ZERO_MIN)
    p.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    p.add_argument('--base-model', default=DEFAULT_BASE_MODEL,
                   help='base model HF mounts as the inference repo')
    p.add_argument('--lora-repo', default=DEFAULT_LORA_REPO,
                   help='LoRA adapter repo passed via TGI LORA_ADAPTERS env')
    p.add_argument('--tgi-image', default=DEFAULT_TGI_IMAGE,
                   help='TGI container image (must support LORA_ADAPTERS env)')
    args = p.parse_args()

    api = HfApi()
    me = api.whoami()
    print(f'HF login: {me["name"]}')

    # BYO-key default: deploy under the logged-in user's namespace so the bill
    # hits THEIR HF account, not the model owner's. Pass --namespace to override
    # (e.g., to deploy under an org you're a member of).
    if args.namespace is None:
        args.namespace = me['name']
        print(f'Deploying under your namespace: {args.namespace} (you pay HF directly)')
    else:
        print(f'Deploying under namespace: {args.namespace}')
        if args.namespace != me['name']:
            org_names = [o['name'] for o in me.get('orgs', [])]
            if args.namespace not in org_names:
                print(f'  WARNING: you are not a member of {args.namespace}. Deploy will fail.')
                print(f'  Your orgs: {org_names or "(none)"}')

    # Check whether an endpoint with this name exists
    try:
        existing = get_inference_endpoint(name=args.name, namespace=args.namespace)
        print(f'Endpoint {args.namespace}/{args.name} already exists '
              f'(status={existing.status}, url={existing.url})')
        if existing.status in ('running', 'scaledToZero', 'initializing', 'updating'):
            print('Reusing existing endpoint. Updating prod_env/model_deployments.yaml…')
            _write_deployments_yaml(existing.url)
            print(f'\nLive URL: {existing.url}')
            print(f'Status:   {existing.status}')
            return 0
        print('  (not in a usable state — will create / re-create after confirmation)')
    except InferenceEndpointError:
        existing = None
    except Exception as e:
        print(f'  status check error: {e}')
        existing = None

    # Pre-flight: confirm the user has Inference Endpoints billing activated.
    # The check is "can I list endpoints in my namespace without a 403?". The
    # 403 hits with a clear "Payment method required" message if not activated.
    try:
        from huggingface_hub import list_inference_endpoints
        list(list_inference_endpoints(namespace=args.namespace, token=api.token))
    except Exception as e:
        if '403' in str(e) or 'Payment method' in str(e):
            print(f'\nERROR: HF Inference Endpoints billing not activated for {args.namespace}.')
            print(f'  Visit https://endpoints.huggingface.co/ in a browser, sign in as')
            print(f'  {args.namespace}, and complete the one-time billing onboarding flow')
            print(f'  (accept ToS + confirm card). Then re-run this script.')
            return 4
        # Other errors (network, etc.) — let the create call surface them

    print('\nProposed endpoint:')
    print(f'  name:                 {args.namespace}/{args.name}')
    print(f'  base model:           {args.base_model}')
    print(f'  LoRA adapter:         {args.lora_repo}  (mounted via TGI LORA_ADAPTERS env)')
    print(f'  instance:             {args.vendor} {args.region} {args.instance_type} ({args.instance_size})')
    print(f'  container:            {args.tgi_image}  (TGI v3, OpenAI-compat /v1/chat/completions)')
    print(f'  scale-to-zero:        after {args.scale_to_zero_min} min idle')
    print()
    print('Cost / quota:')
    print('  HF Inference Endpoint pricing, instance availability, and quota are')
    print('  account-, vendor-, region-, and date-dependent. Verify the current')
    print('  estimate in the HF Endpoint console before creating the endpoint:')
    print('  https://huggingface.co/docs/inference-endpoints/pricing')
    print()

    if not args.yes and not _confirm('Create endpoint? [y/N]'):
        print('Aborted by user. No endpoint created.')
        return 1

    print('\nCreating endpoint with TGI v3 + LoRA adapter…')
    endpoint = api.create_inference_endpoint(
        name=args.name,
        repository=args.base_model,  # base model; TGI loads adapter via env
        framework='pytorch',
        accelerator='gpu',
        instance_type=args.instance_type,
        instance_size=args.instance_size,
        region=args.region,
        vendor=args.vendor,
        task='text-generation',
        min_replica=0,
        max_replica=1,
        scale_to_zero_timeout=args.scale_to_zero_min,
        type='public',  # so users with only the URL can call (still requires HF token in HELM client)
        namespace=args.namespace,
        custom_image={
            'health_route': '/health',
            'url': args.tgi_image,
            'env': {
                'MODEL_ID': '/repository',
                'LORA_ADAPTERS': f'agc-judge={args.lora_repo}',
                'MAX_INPUT_TOKENS': '6144',
                'MAX_TOTAL_TOKENS': '7168',
                'MAX_BATCH_PREFILL_TOKENS': '8192',
            },
        },
    )
    print(f'Created. Polling until running (this can take 5-10 minutes for first deploy)…')

    last_status = None
    deadline = time.time() + 30 * 60  # 30 min cap
    while time.time() < deadline:
        ep = get_inference_endpoint(name=args.name, namespace=args.namespace)
        if ep.status != last_status:
            print(f'  status: {ep.status}')
            last_status = ep.status
        if ep.status == 'running':
            print(f'\nLive URL: {ep.url}')
            _write_deployments_yaml(ep.url)
            print(f'Wrote endpoint URL into {DEPLOYMENTS_YAML}')
            return 0
        if ep.status == 'failed':
            print('\nDEPLOY FAILED. Inspect via HF UI: https://endpoints.huggingface.co/')
            return 2
        time.sleep(20)
    print('\nTIMEOUT after 30 min. Endpoint still initializing — check the HF UI.')
    return 3


def _write_deployments_yaml(url: str):
    """Patch prod_env/model_deployments.yaml in place to point at the live URL.

    Preserve any existing model routes in the file. Users may need those
    routes for the model under evaluation; this helper owns only the
    `agcbench-2026/agc-judge` deployment entry.
    """
    DEPLOYMENTS_YAML.parent.mkdir(parents=True, exist_ok=True)
    new_entry = f"""  - name: agcbench-2026/agc-judge
    model_name: agcbench-2026/AGC-Judge
    tokenizer_name: Qwen/Qwen3-30B-A3B-Instruct-2507
    max_sequence_length: 8192
    client_spec:
      class_name: llm_judge.hf_inference_endpoint_client.HFInferenceEndpointClient
      args:
        base_url: {url}/v1
        openai_model_name: agc-judge
"""
    if not DEPLOYMENTS_YAML.exists():
        header = """# AGC-Judge served via HF Inference Endpoints.
# Autogenerated by eval_scripts/spin_up_hf_endpoint.py — re-run that script
# to refresh the URL after a redeploy.
#
# AGC_JUDGE_OVERRIDE=agcbench-2026/agc-judge resolves to this deployment
# via the resolve_judge() helper in llm_judge/_judge_override.py, then
# HELM dispatches the call through HFInferenceEndpointClient.

model_deployments:
"""
        DEPLOYMENTS_YAML.write_text(header + new_entry)
        return

    text = DEPLOYMENTS_YAML.read_text()
    if 'model_deployments:' not in text:
        text = text.rstrip() + "\n\nmodel_deployments:\n" + new_entry
    else:
        pattern = re.compile(
            r"(?ms)^  - name: agcbench-2026/agc-judge\n.*?(?=^  - name:|\Z)"
        )
        if pattern.search(text):
            text = pattern.sub(new_entry, text)
        else:
            text = text.rstrip() + "\n" + new_entry
    DEPLOYMENTS_YAML.write_text(text)


if __name__ == '__main__':
    sys.exit(main())
