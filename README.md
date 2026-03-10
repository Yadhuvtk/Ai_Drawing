# YD-Vector

YD-Vector is a from-scratch SVG-only foundation model codebase for tokenizer building, dataset indexing, decoder-only transformer training, inference, and export.

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## Pipeline

1. Scan

```powershell
python scripts/00_scan_dataset.py --config configs/data.yaml
```

2. Normalize (optional)

```powershell
python scripts/01_normalize_svgs.py --config configs/data.yaml
```

3. Build index

```powershell
python scripts/02_build_index.py --config configs/data.yaml --out_dir data_local/manifest
```

4. Train

```powershell
python scripts/train.py --config configs/train.yaml
```

5. Infer

```powershell
python scripts/infer.py --config configs/infer.yaml --prompt "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
```

Or:

```powershell
python scripts/infer.py --config configs/infer.yaml --prompt_file .\prompt.svg
```

6. Export

```powershell
python scripts/export.py --run_name yd_vector_small
```

## Windows + CUDA notes

- If CUDA is unavailable or unsupported, code falls back to CPU.
- Reduce `batch_size` or `max_seq_len` if OOM.
- Use lower `num_workers` on Windows if worker startup is slow.

## Default paths

- Raw SVG directory: `E:/Yadhu Projects/SVG`
- Normalized directory: `E:/Yadhu Projects/SVG_NORMALIZED`
- Runs: `outputs/runs/<run_name>`
- Exports: `outputs/exports/<run_name>`

## Build Image->SVG pairs (SVG->PNG)

This step creates PNG<->SVG training pairs by rasterizing SVG files from your manifest with Inkscape CLI.

### 1) Install Inkscape (Windows)

- Download and install Inkscape: https://inkscape.org/release/
- Make sure CLI is available via either:
  - PATH (preferred), or
  - `INKSCAPE_EXE` environment variable

PowerShell examples:

```powershell
# Option A: set explicit Inkscape executable
setx INKSCAPE_EXE "C:\Program Files\Inkscape\bin\inkscape.com"

# Option B: add Inkscape bin to PATH
setx PATH "$($env:PATH);C:\Program Files\Inkscape\bin"
```

Open a new terminal after `setx`.

### 2) Build pairs

Default MVP subset (10,000):

```powershell
python scripts/04_make_im2svg_pairs.py --config configs/im2svg_data.yaml
```

Scale to 100,000:

```powershell
python scripts/04_make_im2svg_pairs.py --config configs/im2svg_data.yaml --limit 100000
```

Run all (`limit=-1`):

```powershell
python scripts/04_make_im2svg_pairs.py --config configs/im2svg_data.yaml --limit -1
```

Useful flags:

- `--size 512` to change raster size
- `--workers 12` to override worker count
- `--overwrite` to regenerate existing PNG files
- `--fresh` to recreate the output manifest file

### 3) Output files

- PNG images: `data_local/renders_256/<sha1_id>.png`
- Pair manifest: `data_local/manifest/im2svg_manifest_256.jsonl`
- Failures report: `data_local/manifest/im2svg_failures_256.jsonl`

### 4) Disk usage and scaling notes

- MVP 10k at 256px usually lands around a few hundred MB to around 1 GB, depending on SVG complexity and PNG compressibility.
- For large runs (100k+ / all), ensure you have substantial free disk and keep `--workers` balanced with CPU and I/O.
- The script is resume-safe: existing PNGs are skipped unless `--overwrite` is used.
