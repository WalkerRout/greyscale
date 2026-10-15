use std::ops::Deref;
use std::sync::{Arc, Mutex};

use wgpu::{
  Backends, Device, Extent3d, Instance, InstanceDescriptor, InstanceFlags, PowerPreference, Queue,
  RequestAdapterOptions, Texture, TextureDescriptor, TextureDimension, TextureFormat,
  TextureUsages,
};

#[derive(thiserror::Error, Debug)]
pub enum GpuError {
  #[error("failed to construct adapter from instance")]
  AdapterCreationFailure(#[from] wgpu::RequestAdapterError),

  #[error("failed to construct logical device from adapter")]
  DeviceCreationFailure(#[from] wgpu::RequestDeviceError),
}

pub struct Scratch {
  texture: Option<Texture>,
  pool: Arc<Mutex<Vec<Texture>>>,
}

impl Deref for Scratch {
  type Target = Texture;

  fn deref(&self) -> &Texture {
    self
      .texture
      .as_ref()
      .expect("scratch texture is only taken on drop")
  }
}

impl Drop for Scratch {
  fn drop(&mut self) {
    if let Some(texture) = self.texture.take() {
      self.pool.lock().unwrap().push(texture);
    }
  }
}

#[derive(Debug, Clone)]
pub struct GpuContext {
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  scratch: Arc<Mutex<Vec<Texture>>>,
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

    Ok(Self {
      device,
      queue,
      scratch: Default::default(),
    })
  }

  pub fn scratch(&self, size: Extent3d, format: TextureFormat) -> Scratch {
    let mut pool = self.scratch.lock().unwrap();
    let texture = match pool
      .iter()
      .position(|t| t.size() == size && t.format() == format)
    {
      Some(index) => pool.swap_remove(index),
      None => self.device.create_texture(&TextureDescriptor {
        label: Some("scratch texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        view_formats: &[],
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
      }),
    };

    Scratch {
      texture: Some(texture),
      pool: self.scratch.clone(),
    }
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
