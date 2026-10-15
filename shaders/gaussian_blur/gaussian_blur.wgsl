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
  
  var total_weight = 0.0;
  var blurred_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  
  for (var dy = -i32(radius); dy <= i32(radius); dy++) {
    for (var dx = -i32(radius); dx <= i32(radius); dx++) {
      let sample_x = min(max(i32(coords.x) + dx, 0), i32(dimensions.x) - 1);
      let sample_y = min(max(i32(coords.y) + dy, 0), i32(dimensions.y) - 1);
      
      let distance_squared = f32(dx * dx + dy * dy);
      let sigma = f32(radius) / 2.0;
      let weight = exp(-distance_squared / (2.0 * sigma * sigma));
      
      total_weight += weight;
      
      let sample_coords = vec2<i32>(sample_x, sample_y);
      let sample_color = textureLoad(input_texture, vec2<u32>(sample_coords), 0);
      blurred_color += sample_color * weight;
    }
  }
  
  blurred_color = blurred_color / total_weight;

  textureStore(output_texture, coords.xy, blurred_color);
}
