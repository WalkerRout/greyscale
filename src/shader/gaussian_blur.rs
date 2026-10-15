use std::mem;

use wgpu::{
  BindingResource, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, Texture,
  TextureViewDescriptor,
};

use crate::gpu::GpuContext;
use crate::shader::{
  Kernel, create_bind_group, create_compute_pipeline, create_shader_module, dispatch,
};

const GAUSSIAN_BLUR_SHADER: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/gaussian_blur/gaussian_blur.wgsl"));

pub struct GaussianBlur {
  pipeline: ComputePipeline,
  context: GpuContext,
  radius_buffer: Buffer,
}

impl GaussianBlur {
  pub fn new(context: &GpuContext, radius: u32) -> Self {
    let shader = create_shader_module(context, "gaussian blur shader", GAUSSIAN_BLUR_SHADER);
    let pipeline = create_compute_pipeline(context, "gaussian blur pipeline", &shader, "main");

    // buffer for blur radius uniform
    let radius_buffer = context.device.create_buffer(&BufferDescriptor {
      label: Some("blur radius buffer"),
      size: mem::size_of::<u32>() as u64,
      usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
      mapped_at_creation: true,
    });

    // write radius value to buffer
    {
      let mut radius_mapped = radius_buffer.slice(..).get_mapped_range_mut();
      // make sure to cap the radius in the shader in case this goes wrong...
      radius_mapped[..4].copy_from_slice(&radius.to_le_bytes());
    }
    // give back to GPU
    radius_buffer.unmap();

    Self {
      pipeline,
      context: context.clone(),
      radius_buffer,
    }
  }
}

impl Kernel for GaussianBlur {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let input_view = input.create_view(&TextureViewDescriptor::default());
    let output_view = output.create_view(&TextureViewDescriptor::default());
    let bind_group = create_bind_group(
      &self.context,
      "gaussian blur bind group",
      &self.pipeline,
      &[
        BindingResource::TextureView(&input_view),
        BindingResource::TextureView(&output_view),
        self.radius_buffer.as_entire_binding(),
      ],
    );
    dispatch(encoder, &self.pipeline, &bind_group, (output.width(), output.height()));
  }
}
