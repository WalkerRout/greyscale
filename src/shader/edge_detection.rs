use wgpu::{
  BindingResource, CommandEncoder, ComputePipeline, Texture, TextureViewDescriptor,
};

use crate::gpu::GpuContext;
use crate::shader::{
  Kernel, create_bind_group, create_compute_pipeline, create_shader_module, dispatch,
};

const EDGE_DETECTION_SHADER: &str = r#"
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1) 
var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let dimensions = textureDimensions(input_texture);
  let coords = vec2<u32>(global_id.xy);
  
  if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
    return;
  }
  
  let sobel_x = mat3x3<f32>(
    -1.0, 0.0, 1.0,
    -2.0, 0.0, 2.0,
    -1.0, 0.0, 1.0
  );
  
  let sobel_y = mat3x3<f32>(
    -1.0, -2.0, -1.0,
    0.0, 0.0, 0.0,
    1.0, 2.0, 1.0
  );
  
  var samples: array<f32, 9>;
  var index = 0;
  
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let sample_x = min(max(i32(coords.x) + dx, 0), i32(dimensions.x) - 1);
      let sample_y = min(max(i32(coords.y) + dy, 0), i32(dimensions.y) - 1);
      
      let sample_coords = vec2<i32>(sample_x, sample_y);
      let sample_color = textureLoad(input_texture, vec2<u32>(sample_coords), 0);
      
      let gray = dot(sample_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
      samples[index] = gray;
      index += 1;
    }
  }
  
  var gx = 0.0;
  var gy = 0.0;
  
  // horizontal gradient
  gx += samples[0] * sobel_x[0][0];
  gx += samples[1] * sobel_x[0][1];
  gx += samples[2] * sobel_x[0][2];
  gx += samples[3] * sobel_x[1][0];
  gx += samples[4] * sobel_x[1][1];
  gx += samples[5] * sobel_x[1][2];
  gx += samples[6] * sobel_x[2][0];
  gx += samples[7] * sobel_x[2][1];
  gx += samples[8] * sobel_x[2][2];
  
  // vertical gradient
  gy += samples[0] * sobel_y[0][0];
  gy += samples[1] * sobel_y[0][1];
  gy += samples[2] * sobel_y[0][2];
  gy += samples[3] * sobel_y[1][0];
  gy += samples[4] * sobel_y[1][1];
  gy += samples[5] * sobel_y[1][2];
  gy += samples[6] * sobel_y[2][0];
  gy += samples[7] * sobel_y[2][1];
  gy += samples[8] * sobel_y[2][2];
  
  let magnitude = sqrt(gx * gx + gy * gy);
  
  let edge_strength = clamp(magnitude / 1.5, 0.0, 1.0);
  
  let original_color = textureLoad(input_texture, coords.xy, 0);
  
  textureStore(output_texture, coords.xy, vec4<f32>(edge_strength, edge_strength, edge_strength, original_color.a));
}
"#;

pub struct EdgeDetection {
  pipeline: ComputePipeline,
  context: GpuContext,
}

impl EdgeDetection {
  pub fn new(context: &GpuContext) -> Self {
    let shader = create_shader_module(context, "edge detection shader", EDGE_DETECTION_SHADER);
    let pipeline = create_compute_pipeline(context, "edge detection pipeline", &shader, "main");

    Self {
      pipeline,
      context: context.clone(),
    }
  }
}

impl Kernel for EdgeDetection {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let input_view = input.create_view(&TextureViewDescriptor::default());
    let output_view = output.create_view(&TextureViewDescriptor::default());
    let bind_group = create_bind_group(
      &self.context,
      "edge detection bind group",
      &self.pipeline,
      &[
        BindingResource::TextureView(&input_view),
        BindingResource::TextureView(&output_view),
      ],
    );
    dispatch(encoder, &self.pipeline, &bind_group, (output.width(), output.height()));
  }
}
