use std::mem;

use wgpu::{
  BindingResource, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline,
  Texture, TextureViewDescriptor,
};

use crate::gpu::GpuContext;
use crate::shader::{
  Kernel, create_bind_group, create_compute_pipeline, create_shader_module, dispatch,
};

const BOX_BLUR_SHADER: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/box_blur/box_blur.wgsl"
));

pub struct BoxBlur {
  pipeline: ComputePipeline,
  context: GpuContext,
  radius_buffer: Buffer,
}

impl BoxBlur {
  pub fn new(context: &GpuContext, radius: u32) -> Self {
    let shader = create_shader_module(context, "box blur shader", BOX_BLUR_SHADER);
    let pipeline = create_compute_pipeline(context, "box blur pipeline", &shader, "main");

    let radius_buffer = context.device.create_buffer(&BufferDescriptor {
      label: Some("blur radius buffer"),
      size: mem::size_of::<u32>() as u64,
      usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
      mapped_at_creation: true,
    });

    {
      let mut radius_mapped = radius_buffer.slice(..).get_mapped_range_mut();
      radius_mapped[..4].copy_from_slice(&radius.to_le_bytes());
    }
    radius_buffer.unmap();

    Self {
      pipeline,
      context: context.clone(),
      radius_buffer,
    }
  }
}

impl Kernel for BoxBlur {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let input_view = input.create_view(&TextureViewDescriptor::default());
    let output_view = output.create_view(&TextureViewDescriptor::default());
    let bind_group = create_bind_group(
      &self.context,
      "box blur bind group",
      &self.pipeline,
      &[
        BindingResource::TextureView(&input_view),
        BindingResource::TextureView(&output_view),
        self.radius_buffer.as_entire_binding(),
      ],
    );
    dispatch(
      encoder,
      &self.pipeline,
      &bind_group,
      (output.width(), output.height()),
    );
  }
}
