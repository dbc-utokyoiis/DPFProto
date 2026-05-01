# DPFProto

## Overview

DPFProto is an GPU-accelerated OLAP query engine prototype that fuses I/O,
decompression, and query execution on the GPU data path. This repository
is the artifact accompanying the PVLDB 2027 submission and allows reviewers
to reproduce the paper's main experimental results on TPC-H and the
Star Schema Benchmark (SSB).

The artifact supports four execution paths (selectable via `-x <mode>`):

| `-x` mode | Description |
|---|---|
| `gidp` | GPU-Initiated Data Path: cuFile (GPUDirect Storage) + host-side decompression |
| `gidp+bam` | GIDP with GPU-initiated I/O via BaM |
| `gidp+bam+fusion` | GIDP+BaM with device-side LZ4 decompression (nvCOMPdx) |
| `datapathfusion` | Full data-path fusion: BaM I/O + PFOR decompression + query scan fused inside a single kernel |

---

## 1. Hardware requirements

The artifact was evaluated on the following machine. A configuration close
to this is required to reproduce the paper's numbers.

| Component | Specification |
|---|---|
| GPU | NVIDIA A100 80 GB PCIe (compute capability 8.0) |
| CPU | Intel Xeon Gold 5418Y, 2 sockets × 24 cores |
| RAM | 503 GiB DDR5-4800 |
| Storage | 4 × NVMe SSD (any PCIe slot; reference system uses `0000:c{0,1,2,3}:00.0`) |

**GPUDirect Storage (GDS)** must be available and working (required by the
`gidp` path). The BaM paths (`gidp+bam`, `gidp+bam+fusion`, `datapathfusion`)
directly bind NVMe queues and therefore require bare-metal NVMe access —
they do **not** work inside containers that hide the NVMe PCI devices.

### 1.1 BIOS settings (required)

DPF follows the same BIOS prerequisites as NVIDIA BaM. See
`bam/README.md` for authoritative details; the settings below are
mandatory for GPU ↔ NVMe peer-to-peer DMA:

| BIOS setting | Required value | Notes |
|---|---|---|
| **Above 4G Decoding** | **Enabled** | Required for the GPU to expose all of its BAR space for P2P. |
| **IOMMU (Intel Vt-d / AMD IOMMU)** | **Disabled** | P2P peer-to-peer over PCIe requires IOMMU off (or passthrough on some systems). |
| **PCIe ACS** | **Disabled** | ACS breaks GPU ↔ NVMe direct DMA. Disable if the BIOS exposes it. |
| **Resizable BAR** | Enabled (recommended) | Helps P2P performance on supported hardware. |
| **PCIe ASPM** | Disabled (recommended) | Reduces I/O latency jitter. |

**Only NVIDIA A100 has been tested end-to-end.** Other GPUs are
unverified.

Ideally the four NVMe devices and the GPU are behind the **same PCIe
switch** (or root complex), so peer-to-peer traffic does not traverse the
IOMMU/CPU. Verify with:

```bash
nvidia-smi topo -m
# Look for PXB / PIX between GPU and the NVMe devices; a SYS link means
# traffic goes through the CPU and throughput will be lower.
```

### 1.2 Kernel boot parameters (required)

The Linux IOMMU support must also be disabled at the kernel level. Edit
`/etc/default/grub` and ensure `GRUB_CMDLINE_LINUX_DEFAULT` does NOT
contain `iommu=on` or `intel_iommu=on`. Add explicit-off flags if needed:

```
GRUB_CMDLINE_LINUX_DEFAULT="... iommu=off intel_iommu=off"
```

Then rebuild GRUB and reboot:

```bash
sudo update-grub
sudo reboot
```

Verify after reboot:

```bash
cat /proc/cmdline | grep -E 'iommu'
# should print iommu=off (or nothing), NOT iommu=on/intel_iommu=on

dmesg | grep -iE 'iommu|dmar' | head
# should show IOMMU disabled / DMAR: IOMMU disabled
```

### 1.3 Identifying the target NVMe devices

### 1.1 Identifying the target NVMe devices

Before partitioning anything, verify that the four data NVMe devices are
the ones expected by the scripts. The artifact assumes:

- **Four physical NVMe devices**, conventionally exposed as
  `/dev/nvme{0,1,2,3}n1`. Any PCIe slot / BDF is acceptable — the BDFs
  `0000:c{0,1,2,3}:00.0` quoted elsewhere in this README are simply what
  the reference machine happens to expose and should **not** be read as
  a hard requirement.
- The OS root/boot device is on a **different** NVMe (e.g. `/dev/nvme4n1`
  at a different PCI address) so the four target devices can be safely
  wiped.
- If your kernel enumerates the data devices under different node names
  (e.g. `/dev/nvme0n1`..`/dev/nvme3n1` hosting the OS on your machine),
  edit `NVME_DEVICES_KERNEL` in `scripts/common.sh` accordingly — no
  other scripts need to be changed.

The helper `scripts/setup/probe_nvme.sh` prints a combined
`/dev/nvmeXn1 ↔ PCI BDF ↔ model ↔ size ↔ root-mount-flag` table in one
go (no sudo required):

```bash
scripts/setup/probe_nvme.sh                    # list all NVMe devices
scripts/setup/probe_nvme.sh -m MZQL21T9        # filter by model substring
```

Raw commands you can also use:

```bash
# List all NVMe namespaces (model, size, namespace id)
sudo nvme list

# Map /dev/nvmeX to its PCI BDF
lspci -d ::0108      # class 0108 = NVMe controller
ls -l /sys/class/nvme/       # shows nvme0 → <pci>/nvme/nvme0

# Confirm which NVMe the OS is currently mounted on (must NOT be touched)
df -h / /boot /home
findmnt / /boot
```

On the reference machine the output looks like:

```
$ sudo nvme list
Node          SN               Model                         Namespace  Usage                       Format           FW Rev
/dev/nvme0n1  S64GNJ0W800461   SAMSUNG MZQL21T9HCJR-00A07    1          1.53 TB / 1.92 TB           512   B +  0 B   GDC5602Q
/dev/nvme1n1  S64GNJ0W800454   SAMSUNG MZQL21T9HCJR-00A07    1          1.38 TB / 1.92 TB           512   B +  0 B   GDC5602Q
/dev/nvme2n1  S64GNJ0W800468   SAMSUNG MZQL21T9HCJR-00A07    1          1.18 TB / 1.92 TB           512   B +  0 B   GDC5602Q
/dev/nvme3n1  S64GNJ0W800469   SAMSUNG MZQL21T9HCJR-00A07    1          1.83 TB / 1.92 TB           512   B +  0 B   GDC5602Q
/dev/nvme4n1  S435NC0T203498   SAMSUNG MZ1LB960HAJQ-00007    1        723.83 GB / 960.20 GB         512   B +  0 B   EDA7602Q
```

```
$ lspci -d ::0108
01:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd ...   ← /dev/nvme4n1 = OS root (do NOT touch)
c0:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd ...   ← /dev/nvme0n1 = data device 0
c1:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd ...   ← /dev/nvme1n1 = data device 1
c2:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd ...   ← /dev/nvme2n1 = data device 2
c3:00.0 Non-Volatile memory controller: Samsung Electronics Co Ltd ...   ← /dev/nvme3n1 = data device 3
```

On this reference machine, the four 1.92 TB data devices happen to be
at `c{0-3}:00.0` (`nvme{0,1,2,3}n1`) and the 960 GB OS drive is at
`01:00.0` (`nvme4n1`). The scripts care about the **kernel node names**
(`NVME_DEVICES_KERNEL` in `scripts/common.sh`), not the PCIe BDFs — any
four data NVMe devices can be used as long as the node names are set
correctly and the OS drive is not among them.

### 1.4 NVMe partition layout (required)

Because DPF uses **NVIDIA BaM** to issue I/O directly from GPU kernels,
the database pages must live on a **raw (unformatted) NVMe partition**
— BaM bypasses the filesystem and binds NVMe submission/completion
queues directly to the block device. At the same time, *source* files
(dbgen output, DuckDB temp files, sideways-pruned CSV, etc.) need a
normal filesystem. The artifact therefore splits every NVMe device into
**two primary partitions** and uses them for different purposes:

- **`/dev/nvmeXn1p1`** — raw partition for BaM-managed database pages.
  The loader (`tpchloader` / `ssbloader`) writes compressed pages
  directly here; the query engine reads them back via BaM.
- **`/dev/nvmeXn1p2`** — assembled into `/dev/md0` (mdadm RAID) and
  mounted at `/export/data1` as XFS. Hosts the filesystem-visible raw
  input data and working directories.

The scripts hard-code this layout and transparently switch between the
two modes by unbinding/rebinding the NVMe driver
(`load_bam` / `unload_bam` in `scripts/common.sh`).

| Partition | Device nodes | Used by | Purpose |
|---|---|---|---|
| **p1** | `/dev/nvme{0,1,2,3}n1p1` | BaM (`gidp+bam`, `gidp+bam+fusion`, `datapathfusion`) and the GIDP loader | Raw block access — the database page layout (compressed columns, zone maps, sideways indexes) is written directly here. |
| **p2** | `/dev/nvme{0,1,2,3}n1p2` | mdadm RAID + XFS | Joined into `/dev/md0` (mdadm RAID over 4 devices) and mounted at `/export/data1` as XFS. Hosts `input${SF}/` (dbgen output), sideways pruning working directories, and compressed archives. |

Partition sizing (reference machine):

- p1: ~800 GiB per device (≈ 3.2 TiB across 4 devices — holds compressed
  database pages for SF up to 400)
- p2: remainder (joined into ~1.6 TiB `/dev/md0` after RAID, large enough
  to stage SF=400 dbgen output and the 278 GiB SF=400 sideways denormalized CSV)

Script hooks (for reference, see `scripts/tpch_run_all.sh:load_bam` /
`unload_bam`):

```bash
# Entering BaM mode (must not be mounted):
sudo umount /export/data1
sudo mdadm --stop /dev/md0
sudo insmod bam/build/module/libnvm.ko

# Leaving BaM mode:
sudo rmmod libnvm
sudo mdadm --assemble /dev/md0 /dev/nvme{0,1,2,3}n1p2
sudo mount -t xfs /dev/md0 /export/data1/
```

Example partitioning (reviewers adapt to their device sizes):

> **⚠ Destructive**: the commands below wipe the NVMe devices. Double-check
> that `/dev/nvme{0,1,2,3}n1` are the **data** devices and *not* the OS
> root, using the `sudo nvme list` / `lspci -d ::0108` / `df -h /`
> commands in §1.1 before running these. If any partition on those
> devices is currently mounted (check `mount | grep /dev/nvme`), `umount`
> it first — `sgdisk` refuses to rewrite the partition table on a device
> whose partitions are in use.

```bash
# For each /dev/nvmeXn1, create two primary partitions.
# Example using sgdisk on a 1.8TB drive:
for dev in /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1; do
    sudo sgdisk --zap-all "$dev"
    sudo sgdisk -n 1:0:+800G -t 1:8300 "$dev"   # p1 = raw (BaM / loader)
    sudo sgdisk -n 2:0:0     -t 2:8300 "$dev"   # p2 = mdadm member
done

# Assemble RAID-0 over the p2 partitions (performance; data is
# regeneratable via dbgen, so no redundancy needed).
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 \
    /dev/nvme{0,1,2,3}n1p2
sudo mkfs.xfs /dev/md0
sudo mkdir -p /export/data1
sudo mount -t xfs /dev/md0 /export/data1
sudo chown "$(whoami):$(whoami)" /export/data1
```

### 1.5 mdadm udev rule workaround (GDS + md RAID)

Per the [NVIDIA GPUDirect Storage Troubleshooting
Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html),
the stock udev rule for md arrays (`/lib/udev/rules.d/63-md-raid-arrays.rules`)
invokes `mdadm --detail` with the `--no-devices` flag. On systems where
GDS / BaM run on top of md RAID this causes race conditions during
array assembly and can block access to the NVMe member devices.

**Fix**: edit `/lib/udev/rules.d/63-md-raid-arrays.rules` so that the
IMPORT line uses the plain `mdadm --detail --export` invocation instead
of the `--no-devices` variant. Open the file with an editor (the file
is owned by root, so use sudo):

```bash
sudo $EDITOR /lib/udev/rules.d/63-md-raid-arrays.rules
# (or: sudo nano /lib/udev/rules.d/63-md-raid-arrays.rules)
```

Locate the IMPORT line near the middle of the file and change it from:

```
IMPORT{program}="/sbin/mdadm --detail --no-devices --export $devnode"
```

to:

```
#IMPORT{program}="/sbin/mdadm --detail --no-devices --export $devnode"
IMPORT{program}="/sbin/mdadm --detail --export $devnode"
```

Verify the edit and reload udev:

```bash
grep '^IMPORT{program}' /lib/udev/rules.d/63-md-raid-arrays.rules
# should print:
#   IMPORT{program}="/sbin/mdadm --detail --export $devnode"

sudo udevadm control --reload-rules
sudo udevadm trigger
```

> **Note**: this edit is to a file under `/lib/udev/rules.d/` and may
> be reverted when the `udev` or `mdadm` package is upgraded. After
> any `apt upgrade` touching those packages, re-apply the edit.

## 2. Software prerequisites

| Component | Version tested |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Linux kernel | 5.15.0-164-generic |
| CUDA Toolkit | 12.8 or later (tested 12.9) |
| GCC / G++ | 11.4.0 |
| CMake | 3.29 or later |
| DuckDB | 1.4.x (used by sideways pruning scripts) |
| Python | 3.11 |
| pzstd | 1.4.x |
| liblz4-dev | any |

Install the system packages:

```bash
sudo apt update
sudo apt install -y \
    liblz4-dev libaio-dev linux-headers-$(uname -r) \
    build-essential cmake curl xz-utils zstd python3 python3-pip \
    gdisk mdadm xfsprogs \
    nvme-cli pciutils
# `zstd` ships `pzstd` (parallel zstd); `gdisk` provides `sgdisk`.
```

**DuckDB CLI** is used by the sideways-pruning helpers
(`scripts/golap/0{1,2}_*.sh`). Install the prebuilt v1.4.4 binary via:

```bash
scripts/setup/install_duckdb.sh              # into /usr/local/bin (uses sudo)
PREFIX=$HOME/.local scripts/setup/install_duckdb.sh   # or user-local
```

The script downloads `duckdb_cli-linux-amd64.zip` from the official
GitHub release and installs the binary as `${PREFIX}/bin/duckdb`.

### 2.1 CUDA Toolkit + GPUDirect Storage (GDS)

Add NVIDIA's CUDA apt repository and install CUDA Toolkit 12.9 together
with the GDS kernel module (`nvidia-fs`). These are the exact packages
used on the reference machine:

```bash
# 1. Add NVIDIA's CUDA apt repository (Ubuntu 22.04 / x86_64)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2. Install CUDA Toolkit 12.9 + GPUDirect Storage
sudo apt install -y cuda-toolkit-12-9 nvidia-gds-12-9
# Reboot might be required

# 3. Add CUDA binaries and libraries to the shell environment
#    (append to ~/.bashrc or equivalent)
export PATH="/usr/local/cuda-12.9/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
```

The `nvidia-gds-12-9` meta-package pulls in `nvidia-fs-dkms` (the GDS
kernel module) and `libcufile0`/`libcufile-dev` (the user-space library).
A driver version ≥ 550 is required; if the host already has an older
driver, `apt` will upgrade it.

### 2.2 Verify the CUDA / GDS installation

After installation (and a reboot if the driver was upgraded):

```bash
nvidia-smi                         # shows A100 80GB and driver version
nvcc --version                     # should report release 12.9
cat /proc/driver/nvidia/version    # confirm driver ≥ 550
lsmod | grep nvidia_fs             # nvidia_fs must be listed (GDS loaded)
ls /dev/nvidia-fs*                 # GDS device nodes exist
/usr/local/cuda/gds/tools/gdscheck.py -p
# runs GDS's sanity check; look for "NVMe : Supported" in the output.
```

### 2.3 cuFile configuration

GDS behaviour is tuned via a JSON configuration read from
`$CUFILE_ENV_PATH_JSON`. DPF ships a preset under `config/cufile.json`
with batch-size and thread-pool values matched to the reference
hardware. The benchmark driver scripts export this variable
automatically, so nothing needs to be done manually in normal use:

```bash
export CUFILE_ENV_PATH_JSON="$(pwd)/config/cufile.json"   # done by tpch_run_all.sh etc.
```

If `gdscheck.py -p` complains about the config or you run `tpchdb`
directly without the wrapper scripts, set `CUFILE_ENV_PATH_JSON`
manually or the GDS library will fall back to `/etc/cufile.json`.

If your NVMe devices are not `/dev/nvme{0,1,2,3}n1`, or the filesystem
layout differs from §3, edit the variables at the top of
`scripts/common.sh` (`NVME_DEVICES_KERNEL`, `MOUNT_POINT`, etc.) once.
Both `tpch_run_all.sh` and `ssb_run_all.sh` source that file, so
changes propagate to every experiment without additional flags.
Each device **must have a `p1` partition** (see §1.4) — the loader
and the BaM path both address `/dev/nvmeXn1p1`; `p1` is not optional.

Optional runtime tuning for reproducible numbers (recommended):

```bash
# Disable transparent hugepages (THP)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
# CPU frequency governor → performance
sudo cpupower frequency-set -g performance
# Reduce interrupt variability: disable cpuidle deep states if not needed
```

## 3. Filesystem layout (fixed)

The artifact scripts assume the following **fixed** mount layout. If your
machine uses different paths, create symlinks before running anything:

```
/export/data1/tpch/       ← TPC-H data root (raw + sideways + archives)
/export/data1/ssb/        ← SSB data root
```

Example for reviewers using a different mount point:

```bash
sudo mkdir -p /export
sudo ln -s /scratch/data /export/data1    # or whichever volume has ≥ 2 TB free
```

## 4. Setup

### 4.1 Clone and fetch submodules

```bash
git clone https://github.com/dbc-utokyoiis/DPFProto DPFProto
cd DPFProto
git submodule update --init --recursive
```

### 4.2 Install NVIDIA redistributables (nvCOMP + nvCOMPdx)

```bash
scripts/setup/install_nvidia_libs.sh
```

This downloads and extracts:

- `nvCOMP 5.1.0.21` (host-side) → `$HOME/libs/nvcomp/`
- `nvidia-mathdx 25.12.1` (device-side nvCOMPdx) → `$HOME/libs/nvidia-mathdx-25.12.1-cuda12/`

By running the script you accept the NVIDIA Software License Agreement
bundled with each archive.

NVIDIA occasionally moves these redistributables to new URLs. If the
script fails with an HTTP 404, manually download the CUDA 12 archives
from the official landing pages and drop them into `$HOME/libs/`:

- nvCOMP   : <https://developer.nvidia.com/nvcomp-downloads>
- nvCOMPdx : <https://developer.nvidia.com/nvcompdx-downloads>

Pick the `nvcomp-linux-x86_64-<ver>_cuda12-archive.tar.xz` and
`nvidia-mathdx-<ver>-cuda12.tar.gz` builds, extract them, and re-create
the `$HOME/libs/nvcomp` symlink that the script would have made.

### 4.3 Build BaM kernel module + userspace library

BaM builds a **kernel module** against your running kernel's headers
and requires the **NVIDIA driver source tree** to be present (the
module is linked against the kernel's NVIDIA driver symbols). Because
the exact prerequisites, kernel-header handling, and secure-boot
caveats depend on the environment, **please follow the official
build instructions in the BaM repository rather than a wrapper here**:

> <https://github.com/ZaidQureshi/bam>  (see `bam/README.md` in this tree)

After the upstream build succeeds, the following artifacts must exist
under the `bam/` submodule checkout — the benchmark scripts locate
them at exactly these paths:

- `bam/build/module/libnvm.ko`  — kernel module (loaded at runtime)
- `bam/build/lib/libnvm.so`     — userspace library (linked by DPF)

The benchmark scripts insmod/rmmod the kernel module automatically
when switching between BaM and XFS modes.

#### Troubleshooting: NVIDIA kernel-module build failures

The BaM build process (re)compiles the NVIDIA kernel modules from the
source tree under `/usr/src/nvidia-<ver>/`. If that step fails,
`bam/README.md` is the authoritative reference. The most common
issue on Ubuntu 22.04 is a compiler mismatch between the running
kernel and the gcc used to build the modules:

```
warning: the compiler differs from the one used to build the kernel
  The kernel was built by: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
  You are using:           gcc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
...
*** Failed CC version check. ***
```

The stock Ubuntu 22.04 kernel is built with gcc-11 but the default
`/usr/bin/gcc` often points to gcc-12. Install gcc-11 and switch the
system default before rebuilding the NVIDIA modules (typically with
`sudo make modules && sudo make modules_install` inside
`/usr/src/nvidia-<ver>/`), then restart the BaM build:

```bash
sudo apt install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
                         --slave   /usr/bin/g++ g++ /usr/bin/g++-11
```

After the NVIDIA modules are built you can restore the previous
default compiler if other tasks in your workflow require a newer gcc.

### 4.4 Build DPF (`tpchdb`, `ssbdb`, loaders)

The build requires **CMake ≥ 3.29.6** (the project uses policy
`CMP0146`, introduced in 3.29.6). Confirm your version before
building:

```bash
cmake --version
```

If you see an error like

```
CMake Error at CMakeLists.txt:65 (cmake_policy):
  Policy "CMP0146" is not known to this version of CMake.
```

install a newer CMake. The official archives (source and prebuilt
Linux binaries) are available at <https://cmake.org/files/> — for
example `cmake-3.29.6-linux-x86_64.tar.gz` under the `v3.29/`
directory. Prepending its `bin/` to `$PATH` is sufficient; no root
install is required.

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Only the four binaries needed by the benchmark scripts to pass compile.
make -j48 tpchdb ssbdb tpchloader ssbloader
cd ..
```

This produces the main binaries:

- `build/tpchdb` — TPC-H query driver
- `build/ssbdb` — SSB query driver
- `build/tpchloader`, `build/ssbloader` — data loaders

### 4.5 Build dbgen (TPC-H) and SSB dbgen

**TPC-H dbgen** is distributed by the TPC under a separate End-User
License Agreement and is **not** bundled with this artifact. Download
the official kit from [tpc.org/tpch](https://www.tpc.org/tpch/), accept
the EULA, build it, and tell the helper scripts where it lives:

The official kit ships a `makefile.suite` template (not a ready-to-use
`Makefile`), so the reviewer has to copy it and set the four top
variables before running `make`:

```bash
# Example: put the kit anywhere you like
cd ~/tpch_kit/dbgen
cp makefile.suite Makefile
# Edit the four top-of-file variables:
#   CC       = gcc
#   DATABASE = INFORMIX       # any supported value works; we don't load
#   MACHINE  = LINUX
#   WORKLOAD = TPCH
make
cd -

# Point the artifact scripts at your built dbgen
export TPCH_DBGEN_DIR=~/tpch_kit/dbgen
# (or uncomment the corresponding line in scripts/common.sh so every
#  shell picks it up automatically)
```

`scripts/tpch/run_dbgen.sh` honours `$TPCH_DBGEN_DIR` and falls back to
`tpch-tool/tpch_2_17_0/dbgen/` only if the variable is unset.

**SSB dbgen** is provided as the `ssb/` submodule in this repository
(upstream: <https://github.com/dbc-utokyoiis/ssb>) and can be built
in place:

```bash
cd ssb
make
cd ..
```

## 5. Data preparation

### 5.1 TPC-H raw data (required SFs: 50, 100, 200, 300, 400)

The helper script `scripts/tpch/run_dbgen.sh` runs dbgen in parallel and
lays out the output tables directly in the per-table directory structure
expected by the loader.

```bash
# Generate a single SF
scripts/tpch/run_dbgen.sh 100

# Generate several SFs at once
scripts/tpch/run_dbgen.sh 50,100,200,300,400

# Override defaults (threads / destination base directory)
scripts/tpch/run_dbgen.sh 100 48 /export/data1/tpch
```

Output layout for SF=100:

```
/export/data1/tpch/input100/
    lineitem/lineitem.tbl.*
    orders/orders.tbl.*
    customer/customer.tbl.*
    part/part.tbl.*
    partsupp/partsupp.tbl.*
    supplier/supplier.tbl.*
    nation/nation.tbl
    region/region.tbl
```

### 5.2 SSB raw data

Use the helper script:

```bash
bash scripts/ssb/run_dbgen.sh 100 48 /export/data1/ssb/input100
# arguments: <SF> <num threads> <output directory>
```

Repeat for SF 50, 200, 300, 400 if the scalability sweep is desired.

### 5.3 Sideways-pruned data (required for zone-map I/O pruning)

For each SF, run the denormalize/sort/split pipeline:

```bash
# TPC-H
bash scripts/golap/01_sideways_pruning.sh -s 100 -n 48

# SSB
bash scripts/golap/02_ssb_sideways_pruning.sh -s 100 -n 48
```

Outputs are written to `/export/data1/tpch/sideways/sf100/` and
`/export/data1/ssb/sideways/sf100/` respectively. The sort keys and rules
are documented in `notes/DENORM.md`.

### 5.4 (Optional) Compressed archives for transport

Sideways data can be archived with pzstd for faster redistribution:

```bash
mkdir -p /export/data1/tpch/sideways_zst
tar -I 'pzstd -p 48' -chf /export/data1/tpch/sideways_zst/sf100.tar.zst \
    -C /export/data1/tpch/sideways sf100
# Expansion in-place on the artifact machine:
tar -I 'pzstd -p 48' -xf /export/data1/tpch/sideways_zst/sf100.tar.zst \
    -C /export/data1/tpch/sideways/
```

`scripts/tpch_run_all.sh` automatically detects and extracts
`sideways_zst/sf${SF}.tar.zst` when present.

## 6. Reproducing the paper's experiments

> **Note for artifact reviewers.** The main entry points are
> `scripts/tpch_run_all.sh` and `scripts/ssb_run_all.sh` (§6.1).
> **Launch them with `sudo`** — they switch the NVMe devices between
> BaM mode and XFS-mounted mode, which requires unbinding/rebinding
> the kernel NVMe driver and unmounting `/export/data1` (see §3).
> The scripts always restore the filesystem on exit (via `trap
> cleanup EXIT`). All driver scripts accept `--dry-run` to print
> commands without running them. Time estimates assume the reference
> hardware in §1.

### 6.1 Main experiment (SF=100, 4 paths × all target queries)

TPC-H Q1, Q3, Q5, Q6, Q13, Q16 and SSB Q1.1–Q4.3 across the four execution
modes:

```bash
# TPC-H
sudo scripts/tpch_run_all.sh -s 100

# SSB
sudo scripts/ssb_run_all.sh -s 100
```

Results are written under:

- `logs/tpch_run_all/<timestamp>/sf100/<mode>/<query>.txt`
- `logs/ssb_run_all/<timestamp>/sf100/<mode>/<query>.txt`

### 6.2 Chunk size sweep

Compression block size sweep on a representative query subset
(TPC-H Q1, Q3, SSB Q1.1, Q2.1, Q3.1) across 64K, 128K, 256K, 512K, 1M, 2M:

```bash
sudo scripts/tpch_chunk_size_sweep.sh -s 100 -c 64K,128K,256K,512K,1M,2M \
    -q q1,q3
sudo scripts/ssb_chunk_size_sweep.sh  -s 100 -c 64K,128K,256K,512K,1M,2M \
    -q q11,q21,q31
```

Results appear under `logs/tpch_chunk_sweep/<timestamp>/` and
`logs/ssb_chunk_sweep/<timestamp>/`.

### 6.3 Scalability sweep

Same representative queries across SF 50, 100, 200, 300, 400:

```bash
sudo scripts/tpch_run_all.sh -s 50,100,200,300,400 -q q1,q3
sudo scripts/ssb_run_all.sh  -s 50,100,200,300,400 -q q11,q21,q31
```

### 6.4 External-systems comparison (SF=100)

Baseline comparisons against DuckDB, Polars, and Spark-RAPIDS are provided
as separate self-contained harnesses under `systems/`:

```bash
# DuckDB
cd systems/duckdb && ./run_bench.sh
# Polars
cd systems/polars && ./run_bench.sh
# Spark-RAPIDS
cd systems/spark-rapids && ./bench_concurrency.sh
```

Each subdirectory has its own README with installation steps and outputs.

### 6.5 (Optional) Correctness verification

Reproducing the paper's **performance** numbers does not require
running these scripts; they are provided so reviewers can check that
the query results themselves match the DuckDB reference answers under
`answers/`:

```bash
# TPC-H: diff each query's result against answers/tpch/sf<N>/<query>.csv
python3 scripts/tpch_verify_answers.py logs/tpch_run_all/<timestamp>

# SSB: row count + aggregated sum match against answers/ssb/sf<N>/
scripts/ssb_verify_bench.sh logs/ssb_run_all/<timestamp>/sf100 100
```

## 7. Output structure

Per-run directories follow this layout:

```
logs/tpch_run_all/YYYYMMDD_HHMMSS/
    main.log
    timings.txt
    sf100/
        gidp/             { q1.txt, q3.txt, q5.txt, q6.txt, q13.txt, q16.txt, load.log }
        gidp+bam/         { ... }
        gidp+bam+fusion/  { ... }
        datapathfusion/   { ... }
        summary.txt
```

Each `q<N>.txt` contains 10 trials with per-trial elapsed time and I/O
statistics; the tail of the file has an `avg / min / max / stddev` summary
line used by `scripts/tpch_verify_answers.py`.

## 8. Command reference (main driver scripts)

Only the most useful flags are shown. Run each script with `-h` or inspect
the script header for the full option list.

**`scripts/tpch_run_all.sh`** — TPC-H across SFs and modes.

| Option | Description |
|---|---|
| `-s SF1,SF2,...` | scale factors (default `50,100,200,300`) |
| `-q q1,q3,...`   | queries (default `q1,q3,q5,q6,q13,q16`) |
| `-n TRIALS`      | number of trials per query (default `10`) |
| `-w THREADS`     | CPU worker threads (default `32`) |
| `-t TIMEOUT`     | per-trial timeout in seconds (default `15`) |
| `--no-zonemap`   | disable zone-map I/O pruning |
| `--skip PHASE`   | skip an execution mode: `gidp`, `gidp+bam`, `gidp+bam+fusion`, `datapathfusion` (repeatable) |
| `--no-load`      | skip data load (data already loaded) |
| `--dry-run`      | print commands without running |

**`scripts/ssb_run_all.sh`** — SSB across SFs and modes (same options).

**`scripts/run_revenue_all.sh`** — Selectivity sweep for the Q6-family
revenue query (TPC-H): fixes `SD_LOW=19920101` and varies `SD_HIGH` from
`19930101` to `19990101` in one-year steps.

## 9. License

This project is released under the MIT License — see [LICENSE](LICENSE).

Third-party components (BaM, FSST, nvCOMP, nvCOMPdx, SSB dbgen,
gpu-compression) retain their original licenses. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.
