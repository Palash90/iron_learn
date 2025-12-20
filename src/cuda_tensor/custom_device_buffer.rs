use crate::{GLOBAL_CONTEXT, GPU_CONTEXT};
use cust::memory::{DeviceBuffer, DevicePointer};

use crate::Numeric;

use core::ffi::c_void;

#[derive(Debug)]
pub struct CustomDeviceBuffer<T: Numeric> {
    pub device_buffer: DeviceBuffer<T>,
}

impl<T: Numeric> CustomDeviceBuffer<T> {
    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        self.device_buffer.as_device_ptr()
    }
}

impl<T: Numeric> Drop for CustomDeviceBuffer<T> {
    fn drop(&mut self) {
        let pool = match &GPU_CONTEXT.get().expect("No GPU Context Set").pool {
            Some(p) => p,
            None => panic!("Cuda not initialized or Gpu Pool is not set up"),
        };

        pool.free(self.as_device_ptr().as_raw());
    }
}

pub fn get_device_buffer<T: Numeric>(size: usize) -> CustomDeviceBuffer<T> {
    let pool = match &GPU_CONTEXT.get().expect("No GPU Context Set").pool {
        Some(p) => p,
        None => panic!("Cuda not initialized or Gpu Pool is not set up"),
    };

    let size = size.checked_mul(size_of::<T>()).unwrap();

    if size == 0 {
        panic!("Attempted a zero size pointer or null pointer creation.");
    }

    let pool_allocated_pointer = pool.allocate(size).unwrap();
    let device_pointer = DevicePointer::from_raw(pool_allocated_pointer);

    let device_buffer = unsafe { DeviceBuffer::from_raw_parts(device_pointer, size) };

    CustomDeviceBuffer { device_buffer }
}

pub fn get_device_buffer_from_slice<T: Numeric>(data: &[T]) -> CustomDeviceBuffer<T> {
    let device_buffer = get_device_buffer::<T>(data.len());
    unsafe {
        cust::sys::cuMemcpyHtoD_v2(
            device_buffer.as_device_ptr().as_raw(),
            data.as_ptr() as *const c_void,
            data.len() * size_of::<T>(),
        )
    };
    device_buffer
}
