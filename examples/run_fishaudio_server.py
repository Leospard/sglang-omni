#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for FishAudio-S1 TTS.

Usage::

    python examples/run_fishaudio_server.py \
        --model-id fishaudio/openaudio-s1-mini \
        --port 8000

Then test with::

    curl http://localhost:8000/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d '{"input": "Hello, how are you?"}' \
        --output output.wav

    # Voice cloning
    curl http://localhost:8000/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d '{"input": "Hello", "ref_audio": "ref.wav", "ref_text": "transcript"}' \
        --output cloned.wav
"""

from __future__ import annotations

import argparse
import logging
import os

from sglang_omni.models.fishaudio_s1 import create_tts_pipeline_config
from sglang_omni.serve import launch_server

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model
    parser.add_argument(
        "--model-id",
        type=str,
        default="fishaudio/openaudio-s1-mini",
        help="Hugging Face model id or local checkpoint path",
    )

    # Device placement
    parser.add_argument(
        "--tts-device",
        type=str,
        default="cuda:0",
        help="GPU device for the TTS engine stage",
    )
    parser.add_argument(
        "--vocoder-device",
        type=str,
        default="cuda:0",
        help="GPU device for the vocoder stage",
    )

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=4096)

    # Compile / cache (on by default)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Disable torch.compile for decode steps",
    )
    parser.add_argument(
        "--no-radix-cache",
        action="store_true",
        default=False,
        help="Disable radix-tree prefix cache",
    )

    # Pipeline options
    parser.add_argument(
        "--relay-type",
        type=str,
        default="shm",
        choices=["shm", "nccl", "nixl"],
        help="Relay type for inter-stage data transfer",
    )

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for /v1/models (default: pipeline name)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = create_tts_pipeline_config(
        model_id=args.model_id,
        tts_device=args.tts_device,
        vocoder_device=args.vocoder_device,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        use_compile=not args.no_compile,
        use_radix_cache=not args.no_radix_cache,
        relay_type=args.relay_type,
    )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
