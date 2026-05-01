# Third-Party Licenses

DPF uses or includes portions of the following third-party components.
The project itself is released under the MIT License (see [LICENSE](LICENSE));
the components below retain their original licenses.

---

## Adapted source (vendored / included in DPF source tree)

### gpu-compression (MIT)

Portions of `src/.../pack.cuh` and related GPU compression kernels are adapted
from Anil Shanbhag's **gpu-compression** project.

- Upstream: https://github.com/anilshanbhag/gpu-compression
- License: MIT (see `notes/gpu-compression/LICENSE`)
- Copyright: (c) 2022 Anil Shanbhag

---

## Git submodules

### BaM — Big accelerator Memory (Simplified BSD / BSD-2-Clause)

- Path: `bam/`
- Upstream: NVIDIA BaM
- License: Simplified BSD (see `bam/LICENSE`)

### FSST — Fast Static Symbol Table (MIT)

- Path: `fsst/` and `fsst-gpu/`
- Upstream FSST: https://github.com/cwida/fsst (CWI Database Architectures group)
- License: MIT (see `fsst/LICENSE`, `fsst-gpu/LICENSE`)

### Star Schema Benchmark dbgen (derivative; see upstream for license)

- Path: `ssb/`
- Upstream: Lab-maintained fork of the Star Schema Benchmark dbgen.
  See `ssb/LICENSE` and `ssb/README.md` for the lineage and license terms.

### CUDA Samples (NVIDIA Software License Agreement)

- Path: `cuda-samples/`
- Upstream: https://github.com/NVIDIA/cuda-samples
- License: See `cuda-samples/LICENSE`

---

## External (downloaded at setup time, not redistributed)

### nvCOMP (NVIDIA Software License Agreement)

- Downloaded by `scripts/setup/install_nvidia_libs.sh` into `$HOME/libs/nvcomp/`
- Upstream: NVIDIA nvCOMP redistributable
- License: NVIDIA Software License Agreement (shipped as `LICENSE` / `NOTICE`
  inside the archive)
- **Not redistributed** in this repository; users agree to the NVIDIA EULA
  when downloading.

### nvidia-mathdx (NVIDIA Software License Agreement)

- Downloaded by `scripts/setup/install_nvidia_libs.sh`
- Upstream: NVIDIA MathDx redistributable (includes nvCOMPdx device API)
- License: NVIDIA Software License Agreement
- **Not redistributed** in this repository.

### CUDA Toolkit and cuFile / GPUDirect Storage (NVIDIA)

- Installed system-wide; not redistributed.
- License: NVIDIA CUDA Toolkit EULA.
