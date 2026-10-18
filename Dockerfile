# syntax=docker/dockerfile:1
#
# Strata: Qwen3.8-Flash-Next in a container. One Dockerfile builds every
# published image; .github/workflows/docker-release.yml passes BASE_IMAGE and
# BACKEND per target, the way the ready-made engine zips are one per backend:
#
#   cuda13  nvidia/cuda:13.0.0-devel-ubuntu24.04  RTX 30/40/50, A-series (sm_75+)
#   cuda12  nvidia/cuda:12.8.1-devel-ubuntu24.04  Pascal sm_60, Volta sm_70, ... -
#           the cards CUDA 13 no longer compiles for (the engine's experimental
#           STRATA_EXPERIMENTAL_SM60 build, docs/OLDER_GPUS.md)
#   hip     rocm/dev-ubuntu-24.04                AMD Radeon gfx1100 / gfx1101 /
#           gfx1200 / gfx1201 (docs/AMD_HIP.md)
#
# The engine is compiled during docker build, so the first container start only
# downloads the model (~70 GB) and starts the server. docker build has no GPU, so
# the architectures are fixed here instead of read from nvidia-smi / rocminfo:
# the engine is a fat binary with code per listed arch, and the runtime picks the
# one matching your card. Narrow the list to your card for a faster build; a card
# outside the set needs a rebuild with its own arch.
#
# Build (CUDA 13, the default):
#   docker build -t strata .
#   docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .        # RTX 40 only
#   docker build -t strata --build-arg STRATA_PORTABLE=1 .            # for another PC
#   docker build -t strata --build-arg STRATA_EXPERIMENTAL_SM60=ON \
#     --build-arg CUDA_ARCHITECTURES=60;70;75 --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04 .
# Build (AMD):
#   docker build -t strata-hip --build-arg BASE_IMAGE=rocm/dev-ubuntu-24.04:7.14.1-full \
#     --build-arg BACKEND=hip --build-arg HIP_ARCHITECTURES=gfx1100 .
#
# STRATA_PORTABLE=1 builds the CPU kernels for an AVX2 baseline instead of the
# processor that builds the image (ggml's -march=native). A local build wants the
# default, 0: it runs on the PC it was built on. A published image wants 1, because
# the machine that builds it is not the one that runs it: a CPU with instruction
# sets the PC lacks (AVX-512 on a CI runner, for one) would otherwise make an image
# that stops there with an illegal instruction. The ready-made engine zips are
# built the same way (tools/hip/build_windows.bat).
#
# Run (NVIDIA host: driver >= 580 for CUDA 13, nvidia-container-toolkit):
#   docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata
# Run (AMD host: the amdgpu driver; the container needs the KFD and DRM devices,
# and the host's video/render groups; docs/AMD_HIP.md):
#   docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render \
#     -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata-hip
#
# Setup choices are env vars, read by docker-entrypoint.sh: FAMILY, MODEL, CONTEXT,
# VISION (no | yes | cpu), KV (int8 | q4_0 | k8v4), GPU (one card) or GPUS ("0,2"
# or "all", with LAYER_SPLIT), LOW_RAM (auto | on | off), HOST, PORT, API_KEY,
# GGUF_DIR (GGUF files you already have), RESIDENT_BUDGET_GIB, KV_STREAMING.
# GPU and GPUS are numbered as nvidia-smi numbers them on NVIDIA, as HIP numbers
# them on AMD.
#
# Only the model files, the prepared pack, the MTP layer and the install config
# live in the /data volume; the engine is part of the image. Strata loads 32-62 GB
# into RAM, so a capped container needs -e LOW_RAM=on: setup.py reads the RAM from
# /proc/meminfo, which here is the host's total, not the container's limit. Add an
# API key before exposing the port to a network: -e API_KEY=<secret>. Pass
# -e REINSTALL=1 to change the model settings later.
#
# --gpus all on a host with two usable cards: setup takes both (the layer split is
# its recommended default). Pin one card with -e GPU=0, or name them with
# -e GPUS=0,2. A volume set up for one card switches to the pair on its first start
# on a two-card host unless GPU or GPUS pins it. LOW_RAM=on runs on one card.

# The toolchain base: CUDA (13, the default, or 12 for older cards) or ROCm.
ARG BASE_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# cuda | hip: which engine this image carries and which toolchain it compiles with.
# The entry point passes it to setup.py, so an AMD image is set up for AMD even on
# a host where detection could pick the other backend.
ARG BACKEND=cuda

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1
# and docker stop's SIGTERM reaches it (see setup.start). Normal Linux starts, which
# don't set it, keep spawning the server as a child.
# LD_LIBRARY_PATH: the ROCm runtime in a ROCm image lives in /opt/rocm/lib (absent
# in a CUDA image, where it costs nothing).
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1 \
    BACKEND=${BACKEND} LD_LIBRARY_PATH=/opt/rocm/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git libatomic1 libgomp1 ninja-build \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

# CUDA: RTX 20 (75), RTX 30 (86), RTX 40 (89), RTX 50 (120), plus 80 for A-series;
# CMakeLists refuses anything below 75 unless STRATA_EXPERIMENTAL_SM60=ON, which the
# CUDA 12 image passes (sm_60 and sm_70 no longer compile with CUDA 13).
# HIP: gfx1100 (RX 7900), gfx1101 (RX 7800/7700), gfx1200 (RX 9060), gfx1201
# (RX 9070 / R9700); docs/AMD_HIP.md lists gfx1030 and gfx1151 as unvalidated.
# BUILD_VISION=0 skips the image encoder; a HIP image builds it for the CPU.
ARG CUDA_ARCHITECTURES=75;80;86;89;120
ARG HIP_ARCHITECTURES=gfx1100;gfx1101;gfx1200;gfx1201
ARG BUILD_VISION=1
ARG STRATA_PORTABLE=0
ARG STRATA_EXPERIMENTAL_SM60=OFF

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

# llama.cpp at the pinned commit, then the engine (and the image encoder), built
# the way setup.py builds them: the same cmake definitions and the same BUILD.json,
# so the first start reuses the engine instead of recompiling. CUDA builds the
# encoder on the GPU (the default); the HIP backend has no GPU encoder yet, so its
# image builds the CPU one (docs/AMD_HIP.md).
RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup

backend = os.environ.get("BACKEND", "cuda")
portable = "ON" if os.environ.get("STRATA_PORTABLE", "0") == "1" else "OFF"   # AVX2 baseline instead of this CPU
want_vision = os.environ.get("BUILD_VISION", "1") == "1"
llama = setup.get_llama_cpp()
common = ["-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
          f"-DSTRATA_PORTABLE={portable}", f"-DSTRATA_GGML_DIR={llama}"]

if backend == "hip":
    root = pathlib.Path("/opt/rocm")
    arch = os.environ.get("HIP_ARCHITECTURES", "gfx1100;gfx1101;gfx1200;gfx1201").strip().strip('"').replace(",", ";")
    libs = [str(root / "lib")]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode")
                    if p.is_dir()), root / "amdgcn" / "bitcode")
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr",
                       "ROCM_PATH": str(root), "HIP_PATH": str(root)})
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*libs, os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"),
                                          os.environ.get("PATH", "")])
    bdir = setup.ROOT / "build-hip"
    setup.cmake_build(setup.ROOT, bdir, "strata",
        ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_PREFILL_MMQ=ON", *common,
         f"-DCMAKE_HIP_ARCHITECTURES={arch}",
         f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}",
         f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
         "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
         f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}"], None, "")
    cuda_dirs, lib_dirs, vision_mode = None, libs, "cpu"
else:
    nvcc, _ = setup.find_nvcc()
    arch = os.environ.get("CUDA_ARCHITECTURES", "75;80;86;89;120").strip().strip('"').replace(",", ";")
    bdir = setup.ROOT / "build"
    setup.cmake_build(setup.ROOT, bdir, "strata",
        ["-DSTRATA_ENABLE_CUDA=ON", *common,
         f"-DSTRATA_EXPERIMENTAL_SM60={os.environ.get('STRATA_EXPERIMENTAL_SM60', 'OFF')}",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"], None, "build-strata.bat")
    bindir = pathlib.Path(nvcc).parent
    cuda_dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
    lib_dirs, vision_mode = None, "gpu"

if want_vision:
    defs = [f"-DLLAMA_DIR={llama}", f"-DSTRATA_VISION_CUDA={'ON' if vision_mode == 'gpu' else 'OFF'}",
            f"-DSTRATA_PORTABLE={portable}"]
    if vision_mode == "gpu":
        defs += [f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
    setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision", defs, None,
                      "build-vision.bat")

eng = setup.ROOT / "engine"
eng.mkdir(exist_ok=True)
shutil.copy2(bdir / setup.EXE, eng / setup.EXE)
if (bdir / setup.KQ_LIB).exists():      # the K-quant MMQ prompt library (UD-Q4/Q5/Q6_K_XL files)
    shutil.copy2(bdir / setup.KQ_LIB, eng / setup.KQ_LIB)
if want_vision:
    shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)

meta = {"version": setup.source_version(), "archs": [x for x in arch.split(";") if x],
        "vision": vision_mode if want_vision else "none",
        "src": setup.source_hash(setup.ENGINE_SOURCES),
        "vision_src": setup.source_hash(setup.VISION_SOURCES) if want_vision else None}
if backend == "hip":
    meta.update({"source": "local-hip", "backend": "hip", "lib_dirs": lib_dirs})
else:
    meta.update({"source": "local", "cuda_dirs": cuda_dirs,
                 "archs": [int(x.split("-")[0]) if x.split("-")[0].isdigit() else x for x in arch.split(";") if x]})
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
PYEOF

# the cmake trees are build-time only; the engine itself is what the container needs
RUN rm -rf build build-vision build-hip

VOLUME ["/data"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one.
# The port only opens after the model loads (1-3 minutes, longer on a first run),
# so the start period is generous: a too short one marks a still-loading container
# unhealthy and a restart policy would kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
