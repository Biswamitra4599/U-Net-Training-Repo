import importlib
import traceback


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_torch_cuda():
    print_header("1) PyTorch CUDA Check")
    try:
        import torch

        print(f"torch.__version__      : {torch.__version__}")
        print(f"CUDA available         : {torch.cuda.is_available()}")
        print(f"torch.version.cuda     : {torch.version.cuda}")
        print(f"CUDA device count      : {torch.cuda.device_count()}")

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            idx = torch.cuda.current_device()
            print(f"Current device index   : {idx}")
            print(f"Current device name    : {torch.cuda.get_device_name(idx)}")

            x = torch.randn(1024, 1024, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print(f"Simple CUDA matmul OK  : {tuple(y.shape)}")
            return True

        print("PyTorch can run, but CUDA is not usable in this environment.")
        return False

    except Exception as exc:
        print("PyTorch CUDA check failed.")
        print(f"Error: {exc}")
        traceback.print_exc()
        return False


def check_cupy():
    print_header("2) CuPy Wheel/Runtime Check")
    try:
        cupy = importlib.import_module("cupy")
        print(f"cupy.__version__       : {cupy.__version__}")

        n_devices = cupy.cuda.runtime.getDeviceCount()
        print(f"CuPy GPU device count  : {n_devices}")

        if n_devices > 0:
            with cupy.cuda.Device(0):
                a = cupy.random.randn(1024, 1024, dtype=cupy.float32)
                b = a @ a
                _ = float(cupy.mean(b).get())
            print("CuPy GPU compute OK    : True")
            return True

        print("CuPy imported, but no GPU devices detected.")
        return False

    except ModuleNotFoundError as exc:
        print("CuPy is not installed or wheel is unavailable for this Python version.")
        print(f"Error: {exc}")
        return False

    except Exception as exc:
        print("CuPy installed but GPU execution failed.")
        print(f"Error: {exc}")
        traceback.print_exc()
        return False


def check_sigpy_gpu(cupy_ok):
    print_header("3) SigPy GPU Check")
    try:
        import sigpy as sp

        print(f"sigpy.__version__      : {sp.__version__}")

        if not cupy_ok:
            print("Skipping SigPy GPU test because CuPy is not usable.")
            print("SigPy GPU path depends on CuPy.")
            return False

        dev = sp.Device(0)
        print(f"SigPy selected device  : {dev}")

        # Minimal SigPy GPU operation
        x = sp.randn((256, 256), device=dev)
        y = sp.fft(x)
        _ = sp.to_device(y, sp.cpu_device)
        print("SigPy GPU FFT OK       : True")
        return True

    except ModuleNotFoundError as exc:
        print("SigPy is not installed.")
        print(f"Error: {exc}")
        return False

    except Exception as exc:
        print("SigPy installed but GPU execution failed.")
        print(f"Error: {exc}")
        traceback.print_exc()
        return False


def summary(torch_ok, cupy_ok, sigpy_ok):
    print_header("Summary")
    print(f"PyTorch CUDA usable    : {torch_ok}")
    print(f"CuPy GPU usable        : {cupy_ok}")
    print(f"SigPy GPU usable       : {sigpy_ok}")

    if sigpy_ok:
        print("\nResult: GPU execution for SigPy/ESPIRiT should be possible in this environment.")
    else:
        print("\nResult: SigPy GPU execution is NOT ready in this environment.")
        print("Likely causes:")
        print("- CuPy wheel not available for your Python version")
        print("- CUDA toolkit/driver mismatch")
        print("- SigPy installed without GPU-capable dependencies")


def main():
    torch_ok = check_torch_cuda()
    cupy_ok = check_cupy()
    sigpy_ok = check_sigpy_gpu(cupy_ok)
    summary(torch_ok, cupy_ok, sigpy_ok)


if __name__ == "__main__":
    main()
