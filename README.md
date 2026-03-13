# YD-Vector

YD-Vector is a local, geometry-driven raster-to-SVG vectorizer project.

This repo is no longer a transformer training or SVG text-generation project. The active codebase is centered on the hybrid vectorizer and its API/backend integration.

## Read This First

If you are a new developer or another AI agent, the most important facts are:

- The active backend route is `POST /api/hybrid-vectorize`
- The active backend code is [app.py](./yd_vector/api/app.py) and [hybrid_service.py](./yd_vector/api/hybrid_service.py)
- The current default vectorization pipeline used by the backend is [pipeline.py](./yd_vector/hybrid_vectorizer/pipeline.py)
- The newer primitive-first experimental path is [pipeline.py](./yd_vector/hybrid_vectorizer_v2/pipeline.py)
- The main CLI entrypoints are [hybrid_vectorize.py](./scripts/hybrid_vectorize.py), [hybrid_vectorize_v2.py](./scripts/hybrid_vectorize_v2.py), and [serve_api.py](./scripts/serve_api.py)
- Old model-training / inference code was removed from source, and should not be considered part of the active architecture

## Current Scope

The repo currently focuses on:

- one-color and flat-color hybrid vectorization
- topology-safe contour extraction
- primitive fitting such as circles, ellipses, rectangles, and rounded rectangles
- freeform curve fitting for non-primitive regions
- local frontend/backend usage
- Windows-friendly offline development

The repo does not currently focus on:

- transformer training
- autoregressive SVG token generation
- model checkpoints / IM2SVG training workflows
- cloud services

## Top-Level Structure

```text
YD-Vector/
  scripts/
    hybrid_vectorize.py
    hybrid_vectorize_v2.py
    serve_api.py
  yd_vector/
    api/
      app.py
      hybrid_service.py
    hybrid_vectorizer/
      ...
    hybrid_vectorizer_v2/
      ...
    utils/
      ...
    paths.py
  outputs/
  data_local/
  inputs/
  README.md
  pyproject.toml
```

## Active Runtime Flow

### Backend / frontend flow

The current generate flow is:

`frontend upload -> POST /api/hybrid-vectorize -> hybrid_service.run_hybrid_vectorization() -> HybridVectorizerPipeline.vectorize() -> outputs/jobs/<job_id>/final.svg`

Key files:

- [app.py](./yd_vector/api/app.py)
- [hybrid_service.py](./yd_vector/api/hybrid_service.py)
- [pipeline.py](./yd_vector/hybrid_vectorizer/pipeline.py)

### CLI flow

Default hybrid vectorizer:

```powershell
python .\scripts\hybrid_vectorize.py --input .\inputs\example.png --output .\outputs\hybrid_vectorizer\example.svg
```

Primitive-first v2:

```powershell
python .\scripts\hybrid_vectorize_v2.py --input .\inputs\example.png --output .\outputs\hybrid_vectorizer_v2\example.svg
```

Local API server:

```powershell
python .\scripts\serve_api.py --host 127.0.0.1 --port 2020
```

## Module Guide

### `yd_vector/api`

- [app.py](./yd_vector/api/app.py)
  - FastAPI app
  - mounts `/outputs`
  - exposes `GET /api/health`
  - exposes `POST /api/hybrid-vectorize`
  - exposes job lookup/download routes

- [hybrid_service.py](./yd_vector/api/hybrid_service.py)
  - saves uploaded input image
  - runs the hybrid vectorizer
  - stores simple in-memory job metadata
  - returns SVG URLs and preview metadata

### `yd_vector/hybrid_vectorizer`

This is the current default vectorizer used by the backend.

- [config.py](./yd_vector/hybrid_vectorizer/config.py)
  - runtime knobs for thresholding, cleanup, primitive fitting, topology tolerances, and export behavior

- [preprocessing.py](./yd_vector/hybrid_vectorizer/preprocessing.py)
  - image loading
  - alpha-aware foreground extraction
  - thresholding
  - basic morphology and small-component cleanup

- [contour_extraction.py](./yd_vector/hybrid_vectorizer/contour_extraction.py)
  - connected components
  - subpixel contour loops
  - hole extraction

- [cleanup.py](./yd_vector/hybrid_vectorizer/cleanup.py)
  - contour simplification
  - corner-aware smoothing
  - narrow-gap preservation

- [shape_analysis.py](./yd_vector/hybrid_vectorizer/shape_analysis.py)
  - circle / ellipse / rectangle / rounded-rectangle fitting helpers
  - arc fitting helpers

- [fitting.py](./yd_vector/hybrid_vectorizer/fitting.py)
  - contour-to-loop fitting
  - primitive promotion
  - freeform segment fitting

- [topology_guard.py](./yd_vector/hybrid_vectorizer/topology_guard.py)
  - validates fitted results against source contours
  - rejects area/bounds/topology distortion

- [svg_export.py](./yd_vector/hybrid_vectorizer/svg_export.py)
  - exports loops and primitives as SVG

- [pipeline.py](./yd_vector/hybrid_vectorizer/pipeline.py)
  - end-to-end pipeline entrypoint

### `yd_vector/hybrid_vectorizer_v2`

This is the newer experimental primitive-first architecture. It is not the default backend path yet.

- [pipeline.py](./yd_vector/hybrid_vectorizer_v2/pipeline.py)
  - main entrypoint for v2

- [structure.py](./yd_vector/hybrid_vectorizer_v2/structure.py)
  - reuses stable preprocessing + contour extraction from the current pipeline

- [decompose.py](./yd_vector/hybrid_vectorizer_v2/decompose.py)
  - prepares contour plans and anchor indices

- [primitive_detection.py](./yd_vector/hybrid_vectorizer_v2/primitive_detection.py)
  - conservative primitive candidate detection

- [ellipse_subshape.py](./yd_vector/hybrid_vectorizer_v2/ellipse_subshape.py)
  - focused lower-base ellipse-like subshape detection

- [freeform.py](./yd_vector/hybrid_vectorizer_v2/freeform.py)
  - freeform fallback fitting for non-primitive spans

- [topology.py](./yd_vector/hybrid_vectorizer_v2/topology.py)
  - wraps the existing topology validator

- [assembler.py](./yd_vector/hybrid_vectorizer_v2/assembler.py)
  - primitive-first loop selection
  - safe fallback assembly

Use v2 when experimenting with:

- lower node count
- primitive-first one-color reconstruction
- special treatment for ellipse-like subshapes such as icon bases

## Which Pipeline Is Actually Used?

At the moment:

- Backend API uses [HybridVectorizerPipeline](./yd_vector/hybrid_vectorizer/pipeline.py)
- Experimental CLI v2 uses [PrimitiveFirstVectorizerV2](./yd_vector/hybrid_vectorizer_v2/pipeline.py)

If you want the backend to switch to v2, the place to change is:

- [hybrid_service.py](./yd_vector/api/hybrid_service.py)

Specifically, `_get_pipeline()` currently builds:

- `HybridVectorizerPipeline(HybridVectorizerConfig())`

## API Contract

### `GET /api/health`

Returns basic backend status and the active generate route.

### `POST /api/hybrid-vectorize`

Accepts:

- multipart file upload named `image`

Returns:

- `ok`
- `job_id`
- `svg_url`
- `input_url`
- `svg_content`
- `preview_steps`

### `GET /api/jobs/{job_id}`

Returns in-memory job metadata for the hybrid vectorization request.

### `GET /api/jobs/{job_id}/svg`

Returns the generated SVG file as `image/svg+xml`.

## Output Layout

Vectorization jobs are written under:

```text
outputs/jobs/<job_id>/
  input.png
  final.svg
```

CLI outputs commonly go to:

```text
outputs/hybrid_vectorizer/
outputs/hybrid_vectorizer_v2/
```

## Local-Only / Ignored Paths

These are local/generated and are ignored by `.gitignore`:

- `outputs/`
- `data_local/`
- `inputs/`
- `runtime/`
- `__pycache__/`
- `*.egg-info/`

Do not treat those as source-of-truth code.

## Legacy / Ignore These If Present

Some folders may still exist locally from earlier repo states or Python caches:

- `yd_vector/data`
- `yd_vector/inference`
- `yd_vector/model`
- `yd_vector/training`
- `configs/`

They are not part of the intended active source architecture anymore. If they only contain stale cache artifacts, ignore them.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## Practical Starting Points

If you are trying to understand or change behavior:

1. Start with [hybrid_service.py](./yd_vector/api/hybrid_service.py) to see what the API runs.
2. Read [pipeline.py](./yd_vector/hybrid_vectorizer/pipeline.py) for the current default flow.
3. Read [pipeline.py](./yd_vector/hybrid_vectorizer_v2/pipeline.py) for the experimental primitive-first flow.
4. For topology issues, inspect [topology_guard.py](./yd_vector/hybrid_vectorizer/topology_guard.py).
5. For primitive fitting quality, inspect [shape_analysis.py](./yd_vector/hybrid_vectorizer/shape_analysis.py) and [primitive_detection.py](./yd_vector/hybrid_vectorizer_v2/primitive_detection.py).

## Short Summary For Another AI

If another AI is given this repo, the correct high-level understanding is:

- This is a hybrid vectorizer repo, not a model-training repo.
- The current backend is FastAPI and only serves hybrid vectorization.
- The default production-like path is `yd_vector.hybrid_vectorizer`.
- The more experimental primitive-first path is `yd_vector.hybrid_vectorizer_v2`.
- The main work should usually happen in those two packages and the API layer, not in any legacy/stale directories.
