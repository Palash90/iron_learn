use cust::memory::DeviceBuffer;
use cust::prelude::*;
use cust::sys::*;
use std::error::Error;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

pub type CUmemoryPool = CUmemPoolHandle_st;
pub type CUmemPoolHandle = *mut CUmemoryPool;

unsafe impl Send for UnsafeCudaMemPoolHandle {}
unsafe impl Sync for UnsafeCudaMemPoolHandle {}

#[derive(Debug)]
struct UnsafeCudaMemPoolHandle(CUmemPoolHandle);

#[derive(Debug)]
pub struct CudaMemoryPool {
    pool: Arc<Mutex<UnsafeCudaMemPoolHandle>>,
}

impl CudaMemoryPool {
    pub fn get_mem_pool() -> CudaMemoryPool {
        let device = Device::get_device(0).unwrap();

        // Create a memory pool for the device
        let mut pool = std::ptr::null_mut();
        let pool_props = CUmemPoolProps {
            allocType: cust::sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: cust::sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: cust::sys::CUmemLocation {
                type_: cust::sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: 0,
            },
            win32SecurityAttributes: std::ptr::null_mut(),
            reserved: [0u8; 64],
        };

        unsafe { cuMemPoolCreate(&mut pool, &pool_props) };

        let reserve_size: usize = 2048 * 1024 * 1024;
        let mut reserve_ptr: CUdeviceptr = 0;
        unsafe {
            // This is often a synchronous call initially, but it gets the memory from the driver
            // and makes it available to the pool.
            cuMemAllocFromPoolAsync(
                &mut reserve_ptr,
                reserve_size,
                pool,
                std::ptr::null_mut(), // Null stream is okay for one-time setup
            );
            // You MUST synchronize the null stream here to ensure memory is available
            cuStreamSynchronize(std::ptr::null_mut());

            // Now free it back to the pool immediately for reuse
            cuMemFreeAsync(reserve_ptr, std::ptr::null_mut());
            cuStreamSynchronize(std::ptr::null_mut());
        }

        println!("Memory pool created for device {}", device.name().unwrap());

        CudaMemoryPool {
            pool: Arc::new(Mutex::new(UnsafeCudaMemPoolHandle(pool))),
        }
    }

    fn with_handle<R, F>(&self, f: F) -> R
    where
        F: FnOnce(CUmemPoolHandle) -> R,
    {
        // Lock the Mutex for exclusive access
        let guard = self.pool.lock().unwrap();

        // The inner raw pointer is the 0th element of the tuple struct
        let raw_handle = guard.0;

        f(raw_handle)
    }

    pub fn allocate(&self, size_in_bytes: usize) -> Result<CUdeviceptr, Box<dyn Error>> {
        let mut device_ptr: CUdeviceptr = 0;
        let byte_size = size_in_bytes;

        self.with_handle(|pool_handle| {
            let now = Instant::now();

            let result = unsafe {
                cuMemAllocFromPoolAsync(
                    &mut device_ptr,
                    byte_size,
                    pool_handle,
                    std::ptr::null_mut(),
                )
            };

            if result != CUresult::CUDA_SUCCESS {
                return Err(format!("CUDA Allocation Failed. Error: {:?}", result).into());
            }

            Ok(device_ptr)
        })
    }

    pub fn free(&self, device_ptr: CUdeviceptr) -> Result<(), Box<dyn Error>> {
        let result = unsafe { cuMemFreeAsync(device_ptr, std::ptr::null_mut()) };

        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("CUDA Free Failed. Error: {:?}", result).into());
        }

        Ok(())
    }
}

impl Drop for CudaMemoryPool {
    fn drop(&mut self) {
        match Arc::try_unwrap(Arc::clone(&self.pool)) {
            Ok(mutex) => {
                match mutex.into_inner() {
                    Ok(unsafe_handle_wrapper) => {
                        let pool_handle = unsafe_handle_wrapper.0;
                        if !pool_handle.is_null() {
                            let result = unsafe { cuMemPoolDestroy(pool_handle) };
                            if result == CUresult::CUDA_SUCCESS {
                                println!(
                                    "Successfully destroyed CUDA Memory Pool: {:?}",
                                    pool_handle
                                );
                            } else {
                                eprintln!(
                                    "WARNING: Failed to destroy CUDA Memory Pool. CUDA Error: {:?}",
                                    result
                                );
                            }
                        }
                    }
                    Err(_) => {
                        // This case should ideally not happen during normal program termination
                        eprintln!("WARNING: Mutex was poisoned during GpuMemoryPool cleanup. Resource may be leaked.");
                    }
                }
            }
            Err(_) => {}
        }
    }
}
