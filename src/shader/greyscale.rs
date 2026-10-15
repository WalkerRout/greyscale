use wgpu::{
  BindingResource, CommandEncoder, ComputePipeline, Texture, TextureViewDescriptor,
};

use crate::gpu::GpuContext;
use crate::shader::{
  Kernel, create_bind_group, create_compute_pipeline, create_shader_module, dispatch,
};

const GRAYSCALE_SHADER: &str = r#"
@group(0) @binding(0)
var input_texture : texture_2d<f32>;

@group(0) @binding(1) 
var output_texture : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id : vec3<u32>,
) {
  let dimensions = textureDimensions(input_texture);
  let coords = vec2<u32>(global_id.xy);

  if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
    return;
  }

  let color = textureLoad(input_texture, coords.xy, 0);
  let grey = dot(vec3<f32>(0.299, 0.587, 0.114), color.rgb);

  textureStore(output_texture, coords.xy, vec4<f32>(grey, grey, grey, color.a));
}
"#;

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
    dispatch(encoder, &self.pipeline, &bind_group, (output.width(), output.height()));
  }
}
