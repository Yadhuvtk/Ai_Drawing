from __future__ import annotations

import hashlib
import io
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from yd_vector.hybrid_vectorizer import HybridVectorizerConfig, HybridVectorizerPipeline
from yd_vector.paths import OUTPUTS_DIR, ensure_dir, relative_repo_path


_HYBRID_JOB_STORE: dict[str, dict[str, Any]] = {}
_HYBRID_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()
_PIPELINE: HybridVectorizerPipeline | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hybrid_busy() -> bool:
    return _HYBRID_LOCK.locked()


def get_hybrid_job(job_id: str) -> dict[str, Any] | None:
    return _HYBRID_JOB_STORE.get(job_id)


def _get_pipeline() -> HybridVectorizerPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = HybridVectorizerPipeline(HybridVectorizerConfig())
    return _PIPELINE


def _save_upload_as_png(image_bytes: bytes, output_path: Path) -> None:
    if not image_bytes:
        raise ValueError("Uploaded image file is empty")

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError("Uploaded file is not a valid image") from exc

    image.load()
    if image.mode not in ("RGB", "RGBA", "LA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    image.save(output_path, format="PNG")


def run_hybrid_vectorization(image_bytes: bytes, image_name: str | None = None) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    started_at = _utc_now()
    _HYBRID_JOB_STORE[job_id] = {"job_id": job_id, "status": "running", "started_at": started_at, "ok": False}

    with _HYBRID_LOCK:
        started = perf_counter()
        try:
            pipeline = _get_pipeline()
            job_dir = ensure_dir(OUTPUTS_DIR / "jobs" / job_id)
            input_path = job_dir / "input.png"
            output_svg_path = job_dir / "final.svg"

            _save_upload_as_png(image_bytes, input_path)
            vectorized = pipeline.vectorize(input_path, output_path=output_svg_path)

            svg_text = vectorized.svg_text
            input_sha1_short = hashlib.sha1(image_bytes).hexdigest()[:12]
            svg_sha1_short = hashlib.sha1(svg_text.encode("utf-8")).hexdigest()[:12]
            duration_ms = round((perf_counter() - started) * 1000.0, 2)

            input_url = f"/outputs/jobs/{job_id}/input.png"
            svg_url = f"/outputs/jobs/{job_id}/final.svg"
            preview_steps = [
                {
                    "id": "original",
                    "name": "Original image",
                    "type": "image",
                    "url": input_url,
                    "durationMs": 700,
                    "presentation": "original",
                    "caption": "Uploaded raster source.",
                },
                {
                    "id": "final",
                    "name": "Final SVG",
                    "type": "svg",
                    "url": svg_url,
                    "svgText": svg_text,
                    "durationMs": 900,
                    "presentation": "final",
                    "caption": "Hybrid vectorizer result.",
                },
            ]

            result = {
                "ok": True,
                "job_id": job_id,
                "status": "completed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_ms": duration_ms,
                "message": "Hybrid vectorization complete",
                "pipeline": vectorized.document.metadata.get("pipeline", "hybrid_vectorizer_monochrome"),
                "image_name": image_name or "uploaded-image",
                "input_url": input_url,
                "svg_url": svg_url,
                "output_dir": relative_repo_path(job_dir),
                "input_path": relative_repo_path(input_path),
                "output_svg_path": relative_repo_path(output_svg_path),
                "svg_content": svg_text,
                "svg_char_length": len(svg_text),
                "image_sha1_short": input_sha1_short,
                "svg_sha1_short": svg_sha1_short,
                "preview_steps": preview_steps,
            }
            _HYBRID_JOB_STORE[job_id] = result
            return result
        except Exception as exc:
            failed = {
                "ok": False,
                "job_id": job_id,
                "status": "failed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "error": str(exc),
            }
            _HYBRID_JOB_STORE[job_id] = failed
            raise
