use wgpu::{
  BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, ComputePipeline, TextureView,
};

use crate::gpu::GpuContext;
use crate::shader::{Kernel, create_compute_pipeline, create_shader_module};

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
      context: GpuContext {
        device: context.device.clone(),
        queue: context.queue.clone(),
      },
    }
  }
}

impl Kernel for Greyscale {
  fn pipeline(&self) -> &ComputePipeline {
    &self.pipeline
  }

  fn create_bind_group(&self, input: &TextureView, output: &TextureView) -> BindGroup {
    self.context.device.create_bind_group(&BindGroupDescriptor {
      label: Some("grayscale bind group"),
      layout: &self.pipeline.get_bind_group_layout(0),
      entries: &[
        BindGroupEntry {
          binding: 0,
          resource: BindingResource::TextureView(input),
        },
        BindGroupEntry {
          binding: 1,
          resource: BindingResource::TextureView(output),
        },
      ],
    })
  }
}
