"""Temporal contract for the bundled 32-step Big Model, not arbitrary models.

Pinned sunnypilot a5f44653, modeld_v2/compile_modeld.py derive_frame_skip:
features_buffer length < 99 uses stride 4 (20 Hz run / 5 Hz context).
"""

BIG_MODEL_FRAME_SKIP = 4
BIG_MODEL_CONTEXT_FRAMES = 33 * BIG_MODEL_FRAME_SKIP
