use crate::GLOBAL_CONTEXT;
use cust::device::Device;
use cust::error::CudaError;
use cust::memory::{DeviceBuffer, DevicePointer};

use crate::Numeric;

use super::mem_pool::CudaMemoryPool;

pub fn get_device_buffer<T: Numeric>(size: usize) -> DeviceBuffer<T> {
    let pool = match &GLOBAL_CONTEXT.get().expect("No Context Set").pool {
        Some(p) => p,
        None => panic!("Cuda not initialized or Gpu Pool is not set up")
    };

    let size = size.checked_mul(size_of::<T>()).unwrap();

    if size == 0 {
        panic!("{}", CudaError::InvalidMemoryAllocation);
    }

    let device_pointer = DevicePointer::from_raw(pool.allocate(size).unwrap());

    unsafe { DeviceBuffer::from_raw_parts(device_pointer, size) }
}
