"""Tear down (delete) the AGC-Judge HF Inference Endpoint.

Usage:
  python eval_scripts/tear_down_hf_endpoint.py [--name NAME] [--namespace NAMESPACE] [--yes]

By default, scales the endpoint to zero (no-op if already there) but does NOT
delete it. Pass --delete to actually remove the endpoint resource (frees the
slot but loses any cached weights, so the next deploy starts from a cold cache).
"""
from __future__ import annotations
import argparse
import sys

from huggingface_hub import HfApi, get_inference_endpoint, InferenceEndpointError


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', default='agc-judge')
    p.add_argument('--namespace', default=None,
                   help='HF namespace (default: your logged-in user)')
    p.add_argument('--delete', action='store_true', help='delete the endpoint resource entirely')
    p.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    args = p.parse_args()

    if args.namespace is None:
        api = HfApi()
        args.namespace = api.whoami()['name']

    try:
        ep = get_inference_endpoint(name=args.name, namespace=args.namespace)
    except InferenceEndpointError:
        print(f'No endpoint named {args.namespace}/{args.name}. Nothing to do.')
        return 0
    print(f'Found {args.namespace}/{args.name}: status={ep.status}, url={ep.url}')

    if args.delete:
        if not args.yes:
            print('Delete this endpoint resource entirely? [y/N]', end=' ', flush=True)
            if sys.stdin.readline().strip().lower() not in ('y', 'yes'):
                print('Aborted.')
                return 1
        ep.delete()
        print(f'Deleted {args.namespace}/{args.name}.')
        return 0

    # Scale to zero (no charge while idle, but resource preserved)
    if ep.status in ('running', 'initializing', 'updating'):
        print('Pausing endpoint (will scale to zero on the next idle window).')
        ep.pause()
    else:
        print(f'Already paused / scaled to zero. No action taken.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
