# DeepIvy: TEE-GPU Hybrid DNN Inference for Model Protection

Source code, NAS search scripts, and end-to-end inference benchmarks for the
ACM CCS 2026 paper on backend-transparent TEE-GPU hybrid DNN inference.

---

## Repository Structure

```
CCS_Artifacts/
├── artifact1_nas/               # Artifact 1: NAS layer-reduction algorithm
│   ├── search.py                # Genetic NAS search driver
│   ├── plot_reduction.py        # Generates Fig. 10 from NAS results
│   ├── Makefile                 # No-op (no native sources)
│   └── run.sh                   # Iterates 3 (model, dataset) pairs
│
├── artifact2_e2e/               # Artifact 2: End-to-end execution framework
│   ├── backbone.py              # Backbone network (untrusted GPU process)
│   ├── tee_server.py            # Enclave-side inference server
│   ├── shm_bridge.cpp           # Zero-copy shared-memory transfer (C++)
│   ├── shmio.py                 # Python wrapper for shared-memory tensors
│   ├── Makefile                 # Builds shm_bridge.so + Gramine SGX manifest
│   ├── run.sh                   # Per-config two-process launch (one fresh pair per cfg)
│   └── gramine/                 # Gramine LibOS manifest template (SGX)
│
├── shared/                      # Common dependencies imported by both artifacts
│   ├── nas/                     # GeneticArchSearchConfig + ArchSearchRunManager
│   ├── modules/                 # MixedEdge / Reduced{Conv,Linear} layers
│   ├── models/lst_vgg/          # LST-VGG (paired with GTSRB)
│   ├── models/lst_resnet/       # LST-ResNet (paired with CIFAR-10)
│   ├── models/lst_vit/          # LST-ViT (paired with CIFAR-100)
│   ├── utils/                   # pytorch_utils, model_deploy, transfer channel, ...
│   └── dataloader.py            # Unified data loader (uses .data/ relative path)
│
├── configs/                     # YAML configs per (model, dataset) pair
├── results/                     # Latency + accuracy outputs (created at runtime)
├── requirements.txt
└── README.md
```

---

## System Requirements

### Hardware

- **Intel SGX2-capable CPU** (e.g., Intel Xeon Ice Lake SP or later) for the
  default `gpu-tee` mode.  If SGX2 is unavailable, the artifact still runs in
  `gpu-cpu` mode (the side server runs as a plain CPU process).
- **NVIDIA CUDA-capable GPU** for the backbone process.

### Operating System

- Ubuntu 20.04 LTS or 22.04 LTS, Linux kernel 5.15+
- **Gramine LibOS v1.8** (only required for `gpu-tee` mode):
  https://gramine.readthedocs.io/en/stable/installation.html

### Software

| Component | Version  |
|-----------|----------|
| GCC       | 11.4+    |
| Python    | 3.10+    |
| PyTorch   | 2.x      |
| CUDA      | 12.x     |
| Gramine   | 1.8 (SGX mode only) |

### Install Python dependencies

**You don't need to do this manually** — both `run.sh` scripts call
`shared/setup_envs.sh`, which creates the two virtualenvs the artifacts need
on first invocation:

- `CCS_Artifacts/my_venv`        — CPU torch wheel (used by `tee_server.py`)
- `CCS_Artifacts/my_venv_cuda`   — CUDA torch wheel (used by `backbone.py` / `search.py`)

Both venvs install `requirements.txt`. Subsequent runs detect the existing
venvs and skip setup.

If you want to provision them by hand instead, the equivalent commands are:

```bash
cd CCS_Artifacts
python3 -m venv my_venv_cuda
my_venv_cuda/bin/pip install --upgrade pip
my_venv_cuda/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
my_venv_cuda/bin/pip install -r requirements.txt

python3 -m venv my_venv
my_venv/bin/pip install --upgrade pip
my_venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
my_venv/bin/pip install -r requirements.txt
```

(Override the wheel index by exporting `TORCH_CUDA_INDEX` / `TORCH_CPU_INDEX`
before running `bash run.sh` if you need a different CUDA version.)

### Datasets

The NAS search reads one batch per (model, dataset) pair to compute the
NASWOT score. Place the datasets under `artifact1_nas/.data/` (the relative
path the loader uses):

| Dataset    | Source                                              |
|------------|-----------------------------------------------------|
| GTSRB      | `torchvision.datasets.GTSRB` will auto-download     |
| CIFAR-10   | `torchvision.datasets.CIFAR10` will auto-download   |
| CIFAR-100  | `torchvision.datasets.CIFAR100` will auto-download  |

The end-to-end latency benchmark (`artifact2_e2e/`) does **not** read any
dataset — it uses dummy inputs to measure inference latency only.

The ViT backbone needs `vit_base_patch16_224` weights. If
`artifact{1,2}_*/models/vit_base_patch16_224.pth` is not present, `timm`
auto-downloads them from HuggingFace on first use.

---

## Reproduce the Results

**Just run the two `run.sh` scripts.** Each one bootstraps the venvs, builds
`shm_bridge.so` (and the SGX manifest, in `gpu-tee` mode), pre-downloads the
ViT weights, and launches the two-process pipeline.

```bash
# Artifact 1: NAS search → produces hall_of_fame.json per (model, dataset)
cd artifact1_nas/
bash run.sh                # default: gpu-tee
# python plot_reduction.py # regenerate Fig. 10 from precomputed/NAS results

# Artifact 2: End-to-end TEE-GPU latency → reproduces Tables 1-2 + Fig. 8
cd ../artifact2_e2e/
bash run.sh                # default: gpu-tee
```

> **`artifact1_nas/run.sh` ships in debug mode** so reviewers can finish a full
> sweep in minutes:
>
> ```bash
> N_GEN=1    # debug mode (paper: 10)
> NPOP=6     # debug mode (paper: 64)
> BETA=30
> ```
>
> For paper-quality NAS results, edit `artifact1_nas/run.sh` and bump
> `N_GEN=10` and `NPOP=64` before running (`BETA=30` is already the paper value).
> A full search takes hours per (model, dataset) pair.

Both scripts iterate the three (model, dataset) configurations reported in the
paper:

- VGG-16   on GTSRB
- ResNet-18 on CIFAR-10
- ViT-Base  on CIFAR-100

`artifact2_e2e/run.sh` launches a **fresh `tee_server.py` + `backbone.py` pair
per config**, runs that one config end-to-end, then tears the pair down before
the next config (running multiple model swaps inside a single SGX enclave can
trip Gramine's malicious-host detection). `backbone.py --cfg <name>` runs a
single config; `--configs_file` is still available to replay NAS-discovered
configs from a JSON list.

Each `run.sh` builds `shm_bridge.so` (and the Gramine SGX manifest, in
`gpu-tee` mode) on first invocation, then launches the two-process pipeline.

### If SGX is not available on your host

Run with the `gpu-cpu` mode — the side server runs as plain Python instead
of a Gramine SGX enclave; everything else is identical:

```bash
cd artifact2_e2e/  &&  bash run.sh gpu-cpu
cd artifact1_nas/  &&  bash run.sh gpu-cpu
```

`gpu-cpu` is also a useful sanity check before attempting `gpu-tee` (Gramine
SGX initialisation can take several minutes).

---

## Outputs

| Artifact    | Output path                                | Contents                                      |
|-------------|--------------------------------------------|-----------------------------------------------|
| Artifact 1  | `artifact1_nas/results/<model>_<dataset>/` | `search.log`, `tee_server.log`, `hall_of_fame.json` |
| Artifact 1  | `artifact1_nas/figures/reduced_layer.pdf`  | Fig. 10 (after running `plot_reduction.py`)   |
| Artifact 2  | `results/<cfg>/backbone.log`               | Per-config latency (ms ± std), one dir per cfg |
| Artifact 2  | `results/<cfg>/tee_server.log`             | Side-process inference + mask/unmask trace    |

Where `<cfg>` is e.g. `lst.8.vgg.gtsrb`, `lst.8.resnet.cifar10`, `lst.8.vit-base.cifar100`.

---

## Compilation

`make` is invoked automatically from `run.sh`; manual invocation:

```bash
cd artifact2_e2e/
make            # compiles shm_bridge.so   (gpu-cpu mode)
make sgx        # also builds pytorch.manifest.sgx + signs (gpu-tee mode)
make clean
```

For `gpu-tee`, the SGX manifest defaults `VENV_DIR` and `PYTHON_BIN` to the
artifact root and the system `python3`. If your venv lives elsewhere, override:

```bash
make sgx VENV_DIR=/abs/path/to/my_venv \
         PYTHON_BIN=/abs/path/to/my_venv/bin/python3
```
