use wgpu::{BindingResource, CommandEncoder, ComputePipeline, Texture, TextureViewDescriptor};

use crate::gpu::GpuContext;
use crate::shader::{
  Kernel, create_bind_group, create_compute_pipeline, create_shader_module, dispatch,
};

const GRAYSCALE_SHADER: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/greyscale/greyscale.wgsl"
));

pub struct Greyscale {
  pipeline: ComputePipeline,
  context: GpuContext,
}

impl Greyscale {
  pub fn new(context: &GpuContext) -> Self {
    let shader = create_shader_module(context, "grayscale shader", GRAYSCALE_SHADER);
    let pipeline = create_compute_pipeline(context, "grayscale pipeline", &shader, "main");
    Self {
      pipeline,
      context: context.clone(),
    }
  }
}

impl Kernel for Greyscale {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let input_view = input.create_view(&TextureViewDescriptor::default());
    let output_view = output.create_view(&TextureViewDescriptor::default());
    let bind_group = create_bind_group(
      &self.context,
      "grayscale bind group",
      &self.pipeline,
      &[
        BindingResource::TextureView(&input_view),
        BindingResource::TextureView(&output_view),
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
