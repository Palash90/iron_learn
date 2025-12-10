import cupy as cp

try:
    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"CuPy/CUDA detected {n_gpus} devices:")
    
    for i in range(n_gpus):
        props = cp.cuda.runtime.getDeviceProperties(i)
        # Decode the name which is returned as a byte string
        print(f"Device ID {i}: {props}")

except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"Error: CuPy cannot see any CUDA devices. Check your NVIDIA drivers and CUDA Toolkit installation. ({e})")