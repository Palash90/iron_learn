#![cfg(feature = "cuda")]
use crate::GPU_CONTEXT;
use cust::memory::{DeviceBuffer, DevicePointer};

use cust::memory::DeviceCopy;

use crate::Numeric;

use core::ffi::c_void;

#[derive(Debug)]
/// Owned wrapper around a CUDA `DeviceBuffer` allocated from the global
/// CUDA memory pool.
///
/// The wrapper ensures memory is returned to the configured `CudaMemoryPool`
/// when dropped. The generic `T` must implement `Numeric` and `DeviceCopy` so
/// it can be safely used across the host/device boundary.
pub struct CustomDeviceBuffer<T: Numeric + DeviceCopy> {
    pub device_buffer: DeviceBuffer<T>,
}

impl<T: Numeric + DeviceCopy> CustomDeviceBuffer<T> {
    /// Return a `DevicePointer<T>` suitable for passing to CUDA kernels or
    /// runtime APIs.
    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        self.device_buffer.as_device_ptr()
    }
}

impl<T: Numeric + DeviceCopy> Drop for CustomDeviceBuffer<T> {
    fn drop(&mut self) {
        let pool = match &GPU_CONTEXT.get().expect("No GPU Context Set").pool {
            Some(p) => p,
            None => panic!("Cuda not initialized or Gpu Pool is not set up"),
        };

        let _ = pool.free(self.as_device_ptr().as_raw());
    }
}

/// Allocate a device buffer of `size` elements from the global CUDA memory
/// pool and return a `CustomDeviceBuffer` that owns the allocation.
///
/// # Panics
/// - If the global `GPU_CONTEXT` is not initialized or the memory pool is
///   not set up.
pub fn get_device_buffer<T: Numeric + DeviceCopy>(size: usize) -> CustomDeviceBuffer<T> {
    let pool = match &GPU_CONTEXT.get().expect("No GPU Context Set").pool {
        Some(p) => p,
        None => panic!("Cuda not initialized or Gpu Pool is not set up"),
    };

    let ptr_size = size.checked_mul(size_of::<T>()).unwrap();

    if ptr_size == 0 {
        panic!("Attempted a zero size pointer or null pointer creation.");
    }

    let pool_allocated_pointer = pool.allocate(ptr_size).unwrap();
    let device_pointer = DevicePointer::from_raw(pool_allocated_pointer);

    let device_buffer = unsafe { DeviceBuffer::from_raw_parts(device_pointer, size) };

    CustomDeviceBuffer { device_buffer }
}

/// Allocate a device buffer and copy the provided `data` slice into device
/// memory, returning a `CustomDeviceBuffer` owning the result.
pub fn get_device_buffer_from_slice<T: Numeric + DeviceCopy>(data: &[T]) -> CustomDeviceBuffer<T> {
    let device_buffer = get_device_buffer::<T>(data.len());
    unsafe {
        cust::sys::cuMemcpyHtoD_v2(
            device_buffer.as_device_ptr().as_raw(),
            data.as_ptr() as *const c_void,
            size_of_val(data),
        )
    };
    device_buffer
}
