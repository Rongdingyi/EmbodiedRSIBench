"""Visual input encoding for the canonical agent (taskbook v0.8, Phase E).

Only `PublicObservation.images` are encoded; no artifact paths, task ids, or
private metadata ever reach the model. Request dumps are redacted by the
accountant.
"""
from __future__ import annotations

import base64
import io

import numpy as np


def image_to_data_url(image) -> str:
    """numpy HWC uint8 or PIL image -> data:image/png;base64,..."""
    try:
        from PIL import Image  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - PIL is a hard dependency
        raise RuntimeError(f"PIL unavailable: {exc}") from exc

    if isinstance(image, Image.Image):
        pil = image
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def observation_image_parts(observation, max_images: int = 2) -> list[dict]:
    """Current public RGB images only; TVR current + target may both enter."""
    parts: list[dict] = []
    for image in list(observation.images or [])[: max(0, int(max_images))]:
        parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
    return parts
