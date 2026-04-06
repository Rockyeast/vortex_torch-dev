from __future__ import annotations

import ast
import json
import os
import shlex
import string
import subprocess
from pathlib import Path
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/vortex_torch"
HF_CACHE_DIR = "/root/.cache/huggingface"
SGLANG_REPO = "https://github.com/dreaming-panda/sglang.git"
SGLANG_BRANCH = os.environ.get("VORTEX_SGLANG_BRANCH", "graph")
SGLANG_SRC_DIR = "/opt/sglang-src"
# setup.py only compiles sm_89 and sm_90, so use Ada/Hopper GPUs by default.
GPU_TYPE = os.environ.get("VORTEX_MODAL_GPU", "L40S")

app = modal.App("vortex-torch")
hf_cache = modal.Volume.from_name("vortex-hf-cache", create_if_missing=True)


def _read_sglang_commit() -> str | None:
    override = os.environ.get("VORTEX_SGLANG_COMMIT")
    if override:
        return override

    if not modal.is_local():
        return None

    try:
        proc = subprocess.run(
            ["git", "submodule", "status", "third_party/sglang"],
            cwd=LOCAL_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None

    line = proc.stdout.strip()
    if not line:
        return None

    sha = line.split()[0].lstrip("-+")
    if len(sha) == 40 and all(ch in string.hexdigits for ch in sha):
        return sha
    return None


def _extract_summary(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            value = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


SGLANG_COMMIT = _read_sglang_commit()

build_commands = [
    "python -m pip install --upgrade pip setuptools wheel ninja",
    "mkdir -p /opt",
    f"rm -rf {SGLANG_SRC_DIR}",
    f"git clone --recursive --branch {shlex.quote(SGLANG_BRANCH)} {shlex.quote(SGLANG_REPO)} {SGLANG_SRC_DIR}",
]

if SGLANG_COMMIT:
    build_commands.append(f"cd {SGLANG_SRC_DIR} && git checkout {SGLANG_COMMIT}")

build_commands.extend(
    [
        f'cd {SGLANG_SRC_DIR} && python -m pip install -e "python[all]"',
        "python -m pip install 'uvloop==0.21.0'",
        "python -m pip install 'flashinfer-python==0.2.7.post1' --no-deps",
        "python -m pip install cachetools watchfiles blake3 py-cpuinfo protobuf "
        "pyyaml python-json-logger 'gguf>=0.13.0' "
        "'lm-format-enforcer>=0.10.11,<0.11' "
        "'prometheus-fastapi-instrumentator>=7.0.0' "
        "'lark==1.2.2' 'xgrammar==0.1.18' "
        "'compressed-tensors==0.9.3' 'depyf==0.18.0'",
        "python -m pip install "
        "'opentelemetry-sdk>=1.26.0,<1.27.0' "
        "'opentelemetry-api>=1.26.0,<1.27.0' "
        "'opentelemetry-exporter-otlp>=1.26.0,<1.27.0' "
        "'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0'",
        "python -m pip install 'mistral_common[opencv]>=1.5.4' "
        "'ray[cgraph]>=2.43.0,!=2.44.*'",
        "python -m pip install 'vllm==0.8.4' --no-deps",
        'python -c "import fastapi, msgspec, multipart, orjson, psutil, setproctitle, uvicorn, uvloop, watchfiles, zmq; import sglang, transformers, vllm; print(\'sglang dependency import check ok\')"',
    ]
)

base_image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "build-essential", "ninja-build", "libnuma1")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "TORCH_CUDA_ARCH_LIST": "8.9;9.0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(*build_commands)
)

app_image = (
    base_image
    .add_local_dir(
        str(LOCAL_ROOT),
        REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".github",
            ".codex",
            "__pycache__",
            "*.pyc",
            "docs",
            "assets/demo.gif",
            "assets/demov2.0.mp4",
            "third_party/sglang",
        ],
    )
    .workdir(REMOTE_ROOT)
    .run_commands(
        "python -m pip install -e . --no-deps --no-build-isolation",
        'python -c "import vortex_torch; import vortex_torch_C; print(\'vortex extension import check ok\')"',
    )
)


@app.function(
    image=app_image,
    gpu=GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=30 * 60,
)
def smoke_test() -> str:
    from sglang.srt.entrypoints.engine import Engine  # type: ignore
    import sglang  # type: ignore
    import torch
    import uvloop
    import vortex_torch
    import vortex_torch_C  # type: ignore

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    device_capability = None
    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_capability = ".".join(str(x) for x in torch.cuda.get_device_capability(0))

    return _to_json(
        {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "sglang": getattr(sglang, "__version__", "unknown"),
        "uvloop": getattr(uvloop, "__version__", "unknown"),
        "vortex_torch": getattr(vortex_torch, "__version__", "unknown"),
        "engine_import_ok": Engine is not None,
        "vortex_extension_ok": vortex_torch_C is not None,
        "cuda_available": torch.cuda.is_available(),
        "device_name": device_name,
        "device_capability": device_capability,
        "nvidia_smi": smi.stdout.strip(),
        "gpu_request": GPU_TYPE,
        "sglang_commit": SGLANG_COMMIT,
        }
    )


@app.function(
    image=app_image,
    gpu=GPU_TYPE,
    volumes={HF_CACHE_DIR: hf_cache},
    timeout=3 * 60 * 60,
)
def run_verify(
    trials: int = 2,
    topk_val: int = 30,
    page_size: int = 16,
    vortex_module_name: str = "gqa_block_sparse_attention",
    model_name: str = "Qwen/Qwen3-1.7B",
    full_attention: bool = False,
    mem: float = 0.8,
) -> str:
    import sys

    cmd = [
        "python",
        "examples/verify_algo.py",
        "--trials",
        str(trials),
        "--topk-val",
        str(topk_val),
        "--page-size",
        str(page_size),
        "--vortex-module-name",
        vortex_module_name,
        "--model-name",
        model_name,
        "--mem",
        str(mem),
    ]
    if full_attention:
        cmd.append("--full-attention")

    proc = subprocess.run(
        cmd,
        cwd=REMOTE_ROOT,
        capture_output=True,
        text=True,
    )

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    proc.check_returncode()
    hf_cache.commit()

    return _to_json(
        {
        "command": " ".join(shlex.quote(part) for part in cmd),
        "summary": _extract_summary(proc.stdout),
        "stdout_tail": proc.stdout[-4000:],
        "gpu_request": GPU_TYPE,
        "sglang_commit": SGLANG_COMMIT,
        }
    )


@app.local_entrypoint()
def main(
    command: str = "check",
    trials: int = 2,
    topk_val: int = 30,
    page_size: int = 16,
    vortex_module_name: str = "gqa_block_sparse_attention",
    model_name: str = "Qwen/Qwen3-1.7B",
    full_attention: bool = False,
    mem: float = 0.8,
) -> None:
    if command == "check":
        result = smoke_test.remote()
    elif command == "verify":
        result = run_verify.remote(
            trials=trials,
            topk_val=topk_val,
            page_size=page_size,
            vortex_module_name=vortex_module_name,
            model_name=model_name,
            full_attention=full_attention,
            mem=mem,
        )
    else:
        raise ValueError("command must be either 'check' or 'verify'")

    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
