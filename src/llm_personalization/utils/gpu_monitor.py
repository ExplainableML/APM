import torch
import subprocess


def log_gpu_usage(tag: str = ""):
    prefix = f"[GPU Monitor] [{tag}]" if tag else "[GPU Monitor]"

    # PyTorch CUDA stats
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            print(
                f"{prefix} GPU {i}: "
                f"allocated={allocated / 1024**3:.1f} GiB, "
                f"reserved={reserved / 1024**3:.1f} GiB, "
                f"free={free / 1024**3:.1f}/{total / 1024**3:.1f} GiB"
            )
    else:
        print(f"{prefix} CUDA not available")

    # System-level nvidia-smi (catches non-PyTorch usage)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            idx, used, total = [x.strip() for x in line.split(",")]
            print(f"{prefix} nvidia-smi GPU {idx}: {used} MiB / {total} MiB used")
    except Exception:
        pass
