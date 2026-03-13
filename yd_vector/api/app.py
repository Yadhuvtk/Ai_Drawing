from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from yd_vector.api.hybrid_service import get_hybrid_job, hybrid_busy, run_hybrid_vectorization
from yd_vector.paths import OUTPUTS_DIR, ensure_dir, resolve_repo_path


app = FastAPI(title="YD-Vector Local API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ensure_dir(OUTPUTS_DIR)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "busy": hybrid_busy(),
        "active_generate_route": "/api/hybrid-vectorize",
        "pipeline": "hybrid_vectorizer",
    }


@app.post("/api/hybrid-vectorize")
async def hybrid_vectorize(image: UploadFile = File(...)) -> dict[str, object]:
    try:
        image_bytes = await image.read()
        return run_hybrid_vectorization(
            image_bytes=image_bytes,
            image_name=image.filename,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, object]:
    result = get_hybrid_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return result


@app.get("/api/jobs/{job_id}/svg")
def job_svg(job_id: str):
    result = get_hybrid_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    if result.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"Job {job_id} is not completed")

    output_path = resolve_repo_path(result["output_svg_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"SVG output missing for job {job_id}")
    return FileResponse(
        path=output_path,
        media_type="image/svg+xml",
        filename=Path(output_path).name,
    )
