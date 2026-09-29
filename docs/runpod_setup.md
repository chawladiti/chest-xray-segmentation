# RunPod setup (A100)

How to stand up the CheXlocalize project on a RunPod A100 pod. This project uses
**isolated per-model environments** because the official model stacks have
conflicting Python/PyTorch requirements.

## On-demand control + automatic on/off (quick reference)

The pod is managed from the local machine with `scripts/runpod.sh` (reads
secrets from git-ignored `.env`: `RUNPOD_API_KEY`, `RUNPOD_POD_ID`). The pod's
public IP/SSH port change on every start and are fetched live — never hard-coded.

```bash
scripts/runpod.sh start      # resume the on-demand A100, wait, print SSH string
scripts/runpod.sh status     # desiredStatus + uptime + $/hr
scripts/runpod.sh info       # current ssh root@IP -p PORT -i KEY
scripts/runpod.sh ssh [cmd]  # shell into the pod (or run a command)
scripts/runpod.sh deploy     # push git-tracked files -> /workspace/chexlocalize-medical-imaging
scripts/runpod.sh autostop   # install pod-side idle auto-stop watchdog
scripts/runpod.sh stop       # stop now (halts GPU billing; keeps /workspace)
```

**Automatic turn-OFF:** `scripts/runpod_idle_autostop.sh` runs *on the pod*,
polls GPU utilization each minute, and calls `runpodctl stop` after
`RUNPOD_IDLE_MINUTES` (default 30) of the GPU staying below
`RUNPOD_IDLE_GPU_UTIL_THRESHOLD`% (default 5). This survives the laptop sleeping,
so an idle A100 stops billing on its own. **Turn-ON is manual/local**
(`scripts/runpod.sh start`) because a stopped pod cannot start itself.

Current pod: on-demand **NVIDIA A100 80GB PCIe**, secure cloud, ~$1.59/hr,
image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`, persistent volume at `/workspace`.

## 0. Prerequisites

1. **No GitHub remote needed.** `scripts/runpod.sh deploy` ships git-tracked
   files over SSH via `git archive` (no data/, no `.venv`, no secrets). You may
   still push to GitHub if you prefer `git clone` on the pod:
   ```bash
   gh repo create <owner>/chexlocalize-medical-imaging --private --source . --push
   ```
2. **Dataset SAS URL** (gated). The CheXlocalize download needs your Stanford AIMI
   SAS URL in `.env` (`CHEXLOCALIZE_SAS_URL`). `.env` is git-ignored and must be
   recreated on the pod (never committed).

## 1. Verify the GPU (run ON the pod)

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Record: Python version · CUDA availability · GPU name · **GPU VRAM** (A100 = 40 GB
or 80 GB) · disk space · free space on `/workspace`.

| Field | Expected (A100 pod) | Actual (fill in) |
| --- | --- | --- |
| GPU | NVIDIA A100 (40/80 GB) | _run nvidia-smi_ |
| CUDA | 12.x | _run_ |
| Python (base) | ≥ 3.10 | _run_ |
| `/workspace` free | tens–hundreds GB | _run `df -h /workspace`_ |

## 2. Clone the repo

```bash
cd /workspace
git clone <your-github-remote-url> chexlocalize-medical-imaging
cd /workspace/chexlocalize-medical-imaging
```

**Repository path on RunPod:** `/workspace/chexlocalize-medical-imaging`
(`/workspace` is RunPod's persistent volume — keep code + data there so they
survive pod restarts). Do **not** use any older/other repository.

## 3. Environment strategy (isolated per model)

The official stacks conflict, so they must **not** share one environment:

| Environment | Python | PyTorch | Purpose |
| --- | --- | --- | --- |
| `core` | ≥ 3.10 | (none needed) | preprocessing, splits, evaluation, tests (this repo's infra) |
| `nnunet` | ≥ 3.10 | recent (CUDA 12) | nnU-Net v2 (trainable) |
| `sam2` | ≥ 3.10 | ≥ 2.5.1 | SAM 2 (foundation) |
| `medsam` | 3.10 | 2.0 | MedSAM (foundation) |

Because the Python **versions differ** (MedSAM pins 3.10, others ≥3.10),
`conda`/`micromamba` is the cleanest choice on RunPod. Structure:

```
/workspace/chexlocalize-medical-imaging/.envs/
├── core/      # infra + evaluation (this repo)
├── nnunet/
├── sam2/
└── medsam/
```

`.envs/` is git-ignored. Create environments **on demand** — do not install every
model stack now. Only the `core` env is needed for the current step:

```bash
# CORE env (infra only — no model stacks)
python -m venv .envs/core            # or: micromamba create -p .envs/core python=3.10
source .envs/core/bin/activate
pip install -r requirements.txt
```

Model envs (created later, each in its own step — **not now**):

```bash
# examples ONLY — do not run yet
# micromamba create -p .envs/sam2   python=3.10   # then torch>=2.5.1 + SAM2
# micromamba create -p .envs/medsam python=3.10   # then torch==2.0 + MedSAM
# micromamba create -p .envs/nnunet python=3.10   # then torch + nnunetv2
```

## 4. Verify Git status & tests (run ON the pod, in `core`)

```bash
git status
git log --oneline | head -5
pytest -q                       # expect: 71 passed
python scripts/check_environment.py
```

## 5. Project configuration access

The central config loads without any model deps:

```bash
python scripts/validate_project.py           # skeleton + config sanity
python -c "from configs.loader import load_config; c=load_config(); \
print(c.name, c.mode, c.paths['raw_data_root'])"
```

## 6. Dataset location on RunPod

Keep the dataset on the persistent volume, under the repo's `data/` tree (which is
git-ignored):

- **Raw dataset:** `/workspace/chexlocalize-medical-imaging/data/raw/chexlocalize/`
- **Processed masks/targets:** `/workspace/chexlocalize-medical-imaging/data/processed/`
- **Splits (committed):** `/workspace/chexlocalize-medical-imaging/data/splits/`

Download on the pod (after setting `CHEXLOCALIZE_SAS_URL` in `.env`):

```bash
cp .env.example .env      # then add CHEXLOCALIZE_SAS_URL=...
bash scripts/download_chexlocalize.sh
python scripts/verify_dataset.py
python scripts/preprocess.py
python scripts/build_splits.py
```

Alternatively, if the dataset already exists elsewhere on the pod, point the paths
via env vars (no copying):

```bash
export RAW_DATA_ROOT=/workspace/datasets/chexlocalize
```

## 7. Environment verification commands (summary)

```bash
nvidia-smi
python scripts/check_environment.py
python scripts/validate_project.py
pytest -q
```

## What is intentionally NOT done here

- No training, no model-checkpoint downloads.
- SAM 2 / MedSAM / nnU-Net **not** installed.
- Dataset files not modified. Methodology unchanged.
