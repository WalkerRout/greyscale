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
