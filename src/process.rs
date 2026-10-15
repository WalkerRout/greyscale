use std::mem;

use image::{ImageBuffer, Rgba};

use wgpu::{
  BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode, PollType,
  TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
  TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

use crate::gpu::{self, GpuContext};
use crate::shader::Kernel;

pub type Image = ImageBuffer<Rgba<u8>, Vec<u8>>;

#[derive(thiserror::Error, Debug)]
pub enum ProcessError {
  #[error("failed to poll device queue")]
  DevicePollFailure(#[from] wgpu::PollError),

  #[error("failed to construct an image given output data")]
  ImageCreationFailure,
}

pub struct ImageProcessor {
  context: GpuContext,
}

impl ImageProcessor {
  pub fn new(ctx: &GpuContext) -> Self {
    Self {
      context: ctx.clone(),
    }
  }

  pub fn process_image<K>(&self, image: &Image, kernel: &K) -> Result<Image, ProcessError>
  where
    K: for<'a> Kernel<Input<'a> = &'a Texture>,
  {
    let (width, height) = image.dimensions();

    let extent = Extent3d {
      width,
      height,
      depth_or_array_layers: 1,
    };

    // create input texture
    let input_texture = self.context.device.create_texture(&TextureDescriptor {
      label: Some("input texture"),
      size: extent,
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba8Unorm, // convert from/to [0, 255] to/from [0.0, 1.0] in shader
      view_formats: &[TextureFormat::Rgba8Unorm],
      usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    });

    // upload image data to the image texture
    self.context.queue.write_texture(
      input_texture.as_image_copy(),
      bytemuck::cast_slice(image.as_raw()),
      TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(4u32.saturating_mul(width)),
        rows_per_image: None,
      },
      extent,
    );

    // create output texture
    let output_texture = self.context.device.create_texture(&TextureDescriptor {
      label: Some("output texture"),
      size: extent,
      mip_level_count: 1,
      sample_count: 1,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba8Unorm,
      view_formats: &[TextureFormat::Rgba8Unorm],
      // src because we copy it to buffer; shader can modify since we use storage_binding
      usage: TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
    });

    // determine how many bytes each row has, with/without padding
    let padded_bytes_per_row = gpu::padded_bytes_per_row(width);
    let unpadded_bytes_per_row = width as usize * 4;
    // final buffer size based on padded data
    let output_buffer_size =
      padded_bytes_per_row as u64 * height as u64 * mem::size_of::<u8>() as u64;

    // create a buffer to hold output texture data
    let output_buffer = self.context.device.create_buffer(&BufferDescriptor {
      label: Some("output buffer"),
      size: output_buffer_size,
      usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
      mapped_at_creation: false,
    });

    let mut encoder = self
      .context
      .device
      .create_command_encoder(&CommandEncoderDescriptor { label: None });

    kernel.encode(&mut encoder, &input_texture, &output_texture);

    encoder.copy_texture_to_buffer(
      TexelCopyTextureInfo {
        texture: &output_texture,
        mip_level: 0,
        origin: Default::default(),
        aspect: TextureAspect::All,
      },
      TexelCopyBufferInfo {
        buffer: &output_buffer,
        layout: TexelCopyBufferLayout {
          offset: 0,
          bytes_per_row: Some(padded_bytes_per_row as u32),
          rows_per_image: Some(height),
        },
      },
      extent,
    );

    let submission_index = self.context.queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(MapMode::Read, |_| {});

    self
      .context
      .device
      .poll(PollType::WaitForSubmissionIndex(submission_index))
      .map_err(ProcessError::DevicePollFailure)?;

    let padded_data = buffer_slice.get_mapped_range();
    let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
    for (padded, pixels) in padded_data
      .chunks_exact(padded_bytes_per_row)
      .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
    {
      pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
    }

    Image::from_raw(width, height, pixels).ok_or(ProcessError::ImageCreationFailure)
  }
}
