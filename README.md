# DeepIvy: TEE-GPU Hybrid DNN Inference for Model Protection

**Author:** Reyad Islam

Source code, NAS search scripts, and end-to-end inference benchmarks for the ACM CCS 2026 paper on backend-transparent TEE-GPU hybrid DNN inference.

---

## Repository Structure

```
repoladder/
├── artifact1_nas/               # Artifact 1: NAS layer-reduction algorithm
│   ├── search.py                # Genetic NAS search driver
│   ├── plot_reduction.py        # Post-search figure generation script
│   ├── Makefile                 # Local build rules
│   ├── run.sh                   # Experiment runner
│   └── results/
│       └── precomputed/         # Pre-computed results for plotting
│
├── artifact2_e2e/               # Artifact 2: End-to-end execution framework
│   ├── backbone.py              # Backbone network (untrusted GPU process)
│   ├── tee_server.py            # Enclave-side inference server
│   ├── shm_bridge.cpp           # Zero-copy shared-memory transfer layer (C++)
│   ├── shmio.py                 # Python wrapper for shared-memory tensors
│   ├── Makefile                 # Compiles shm_bridge.cpp
│   ├── run.sh                   # Coordinates two-process launch
│   └── gramine/                 # Gramine LibOS manifests for SGX enclave
│
├── configs/                     # YAML configuration files (model + dataset pairs)
├── results/                     # Collected latency and accuracy outputs
├── requirements.txt             # Python dependencies
└── README.md
```

---

## System Requirements

### Hardware

- An **Intel SGX2-capable CPU** (e.g., Intel Xeon Ice Lake SP or later)
- An **NVIDIA CUDA-capable GPU**

The backbone process (GPU) and TEE-side server (SGX enclave) run as co-located processes on the same host and communicate through a POSIX shared memory segment.

### Operating System

- Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
- Linux kernel 5.15 or later
- Intel SGX in-kernel driver must be installed
- **Gramine LibOS v1.8** must be installed — see [installation instructions](https://gramine.readthedocs.io/en/stable/installation.html)

### Software Dependencies

| Component | Version |
|-----------|---------|
| GCC | 11.4 or later |
| Python | 3.10 or later |
| PyTorch | 2.x |
| CUDA | 12.x |
| Gramine | 1.8 |

Install all Python dependencies inside a virtual environment:

```bash
pip install -r requirements.txt
```

---

## Provided Artifacts

### Artifact 1: NAS Algorithm

Reproduces the layer-reduction figure (Fig. 3 in the paper).

**Directory:** `artifact1_nas/`

| File | Description |
|------|-------------|
| `search.py` | Genetic NAS search driver; explores layer-reduction design space per model–dataset pair |
| `plot_reduction.py` | Generates the figure from collected or pre-computed results |
| `results/precomputed/` | Pre-computed results; use directly to skip the search |

### Artifact 2: End-to-End Execution Framework

Reproduces main latency and accuracy results (Tables 1–2 and Fig. 5 in the paper).

**Directory:** `artifact2_e2e/`

| File | Description |
|------|-------------|
| `backbone.py` | Backbone network executed on the untrusted GPU |
| `tee_server.py` | Enclave-side inference server; manages request queuing and shared-memory handshakes |
| `shm_bridge.cpp` | Zero-copy tensor transfer layer (C++); implements POSIX SHM with spinlock-based atomic producer-consumer protocol |
| `shmio.py` | Python wrapper; maps SHM pointer as NumPy array and exposes it as a `torch.Tensor` |
| `gramine/` | Gramine manifests for running `tee_server.py` inside an SGX enclave |

---

## Compilation

C/C++ sources (`shm_bridge.cpp`) are compiled using the provided `Makefile`. Each artifact subdirectory contains a local `Makefile` for individual compilation.

**Compile Artifact 2 shared-memory bridge:**

```bash
cd artifact2_e2e/
make
```

**Compile all artifacts from the root:**

```bash
make -C artifact1_nas/
make -C artifact2_e2e/
```

---

## Configuration

Edit the relevant `.yaml` files in `configs/` to specify the model family and dataset. Supported model–dataset pairs match those evaluated in the paper (CNNs and attention-based models).

```bash
# Example: select ResNet-50 on CIFAR-100
configs/resnet50_cifar100.yaml
```

---

## Execution

Each artifact directory contains a `run.sh` script that executes the experiment with recommended parameters.

### Artifact 1: NAS Search

```bash
cd artifact1_nas/
bash run.sh
# To skip search and plot from pre-computed results:
python plot_reduction.py --results results/precomputed/
```

### Artifact 2: End-to-End Inference

**Start the enclave server first:**

```bash
cd artifact2_e2e/
gramine-sgx tee_server.py
```

Once the enclave prints `[TEE] ready`, launch the GPU backbone in a second terminal:

```bash
python backbone.py
```

The `run.sh` script automates this two-process coordination and collects latency and accuracy into `results/`:

```bash
bash run.sh
```

---

## Results

Output files are written to `results/` in each artifact directory. Latency (ms) and accuracy (%) are logged per model–dataset pair and match the values reported in the paper tables.
