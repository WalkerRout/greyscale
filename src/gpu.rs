use std::sync::Arc;

use wgpu::{
  Backends, Device, Instance, InstanceDescriptor, InstanceFlags, PowerPreference, Queue,
  RequestAdapterOptions,
};

#[derive(thiserror::Error, Debug)]
pub enum GpuError {
  #[error("failed to construct adapter from instance")]
  AdapterCreationFailure(#[from] wgpu::RequestAdapterError),

  #[error("failed to construct logical device from adapter")]
  DeviceCreationFailure(#[from] wgpu::RequestDeviceError),
}

#[derive(Debug, Clone)]
pub struct GpuContext {
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
}

impl GpuContext {
  pub async fn new() -> Result<Self, GpuError> {
    let instance = Instance::new(&InstanceDescriptor {
      backends: Backends::all(),
      flags: InstanceFlags::DEBUG,
      backend_options: Default::default(),
    });

    let adapter = instance
      .request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
      })
      .await
      .map_err(GpuError::AdapterCreationFailure)?;

    let (device, queue) = adapter
      .request_device(&Default::default())
      .await
      .map_err(GpuError::DeviceCreationFailure)?;

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    Ok(Self { device, queue })
  }
}

pub fn compute_work_group_count(
  (width, height): (u32, u32),
  (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
  let x = width.div_ceil(workgroup_width);
  let y = height.div_ceil(workgroup_height);
  (x, y)
}

pub fn padded_bytes_per_row(width: u32) -> usize {
  let bytes_per_row = width as usize * 4;
  let padding = (256 - bytes_per_row % 256) % 256;
  bytes_per_row + padding
}
