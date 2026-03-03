# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro (DualAR / fish_qwen3_omni) model support for sglang-omni.

The S2-Pro model uses the same DualAR architecture as S1-Mini, so all
runtime, IO, and tokenizer components are reused from ``fishaudio_s1``.
Only the pipeline configuration differs (default model ID, pipeline name).
"""

from sglang_omni.models.fishaudio_s1.factory import create_dual_ar_engine
from sglang_omni.models.fishaudio_s1.runtime.dual_ar import (
    DualARRequestData,
    DualARStepOutput,
)
from sglang_omni.models.fishaudio_s1.tokenizer import FishTokenizerAdapter, Reference

__all__ = [
    "create_dual_ar_engine",
    "create_tts_pipeline_config",
    "DualARRequestData",
    "DualARStepOutput",
    "FishTokenizerAdapter",
    "Reference",
]


def __getattr__(name: str):
    if name == "create_tts_pipeline_config":
        from .pipeline.config import create_tts_pipeline_config

        return create_tts_pipeline_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
