# SPDX-License-Identifier: Apache-2.0
"""Stage routing callbacks for the FishAudio S2-Pro TTS pipeline."""

from sglang_omni.models.fishaudio_s1.pipeline.next_stage import (  # noqa: F401
    PREPROCESSING_STAGE,
    TTS_ENGINE_STAGE,
    VOCODER_STAGE,
    preprocessing_next,
    tts_engine_next,
    vocoder_next,
)
