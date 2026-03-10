from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from yd_vector.api.service import current_device_name, get_job, is_busy, list_inference_configs, resolve_repo_path, run_image_inference


class InferRequest(BaseModel):
    config_path: str
    image_data_url: str
    image_name: str | None = None
    prompt: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


app = FastAPI(title="YD-Vector Local API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    configs = list_inference_configs()
    return {
        "status": "ok",
        "device": current_device_name(),
        "busy": is_busy(),
        "config_count": len(configs),
    }


@app.get("/api/configs")
def configs() -> dict[str, object]:
    configs = list_inference_configs()
    return {"configs": configs}


@app.post("/api/infer")
def infer(request: InferRequest) -> dict[str, object]:
    try:
        return run_image_inference(
            config_path=request.config_path,
            image_data_url=request.image_data_url,
            image_name=request.image_name,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, object]:
    result = get_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return result


@app.get("/api/jobs/{job_id}/svg")
def job_svg(job_id: str):
    result = get_job(job_id)
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
