"""Check that PyTorch can see your NVIDIA GPU before running the experiments.

Usage:  python scripts/check_gpu.py
"""
import platform
import subprocess
import sys
import time


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. See LAPTOP_RUNBOOK.md, step 2.")
        return 1

    print(f"PyTorch : {torch.__version__}   built with CUDA: {torch.version.cuda}")
    if torch.version.cuda is None:
        print("\nPROBLEM: this is the CPU-only build of PyTorch.")
        if platform.system() == "Windows":
            print("On Windows, plain `pip install torch` gives the CPU build. Uninstall it and reinstall the CUDA build:")
        else:
            print("Reinstall the CUDA build:")
        print("  pip uninstall -y torch torchvision")
        print("  then copy the install command for your OS + 'CUDA' from https://pytorch.org/get-started/locally/")
        return 1
    if not torch.cuda.is_available():
        print("\nPROBLEM: PyTorch has CUDA support but cannot see a GPU.")
        print("  - Update the NVIDIA driver (GeForce Experience / NVIDIA App, or nvidia.com/drivers).")
        print("  - Run `nvidia-smi` in a terminal; it must list the RTX 4050.")
        print("  - On Windows laptops, set python.exe to 'High performance' in Settings > System > Display > Graphics.")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"GPU     : {props.name}   {props.total_memory / 2**30:.1f} GB   compute capability {props.major}.{props.minor}")

    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    for _ in range(3):
        x @ x
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        x @ x
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096 ** 3 / (time.perf_counter() - t) / 1e12
    print(f"FP32 matmul: {tflops:.1f} TFLOPS (rough; a laptop RTX 4050 should show several TFLOPS on AC power)")

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm",
                              "--format=csv"], capture_output=True, text=True, timeout=10).stdout
        print("\nnvidia-smi:\n" + out.strip())
    except Exception:
        print("\n(nvidia-smi not found on PATH; that's OK, PyTorch sees the GPU.)")
    print("\nGPU OK. Next: python scripts/download_weights.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
