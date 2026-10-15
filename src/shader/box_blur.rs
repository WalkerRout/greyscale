use std::mem;

use wgpu::{
  BindingResource, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, Texture,
  TextureViewDescriptor,
};

use crate::gpu::GpuContext;
use crate::shader::{
  Kernel, create_bind_group, create_compute_pipeline, create_shader_module, dispatch,
};

const BOX_BLUR_SHADER: &str = r#"
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1) 
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> blur_radius: u32;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let dimensions = textureDimensions(input_texture);
  let coords = vec2<u32>(global_id.xy);
  
  if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
    return;
  }
  
  let radius = min(blur_radius, 15u);
  
  var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var count = 0;
  
  for (var dy = -i32(radius); dy <= i32(radius); dy++) {
    for (var dx = -i32(radius); dx <= i32(radius); dx++) {
      let sample_x = min(max(i32(coords.x) + dx, 0), i32(dimensions.x) - 1);
      let sample_y = min(max(i32(coords.y) + dy, 0), i32(dimensions.y) - 1);
      
      let sample_coords = vec2<i32>(sample_x, sample_y);
      let sample_color = textureLoad(input_texture, vec2<u32>(sample_coords), 0);
      sum += sample_color;
      count += 1;
    }
  }
  
  let avg_color = sum / f32(count);
  
  textureStore(output_texture, coords.xy, avg_color);
}
"#;

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
    dispatch(encoder, &self.pipeline, &bind_group, (output.width(), output.height()));
  }
}
