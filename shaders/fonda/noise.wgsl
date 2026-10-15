@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { seed: f32 }
@group(0) @binding(2) var<uniform> p: Params;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn unit(h: u32) -> f32 {
  return (f32(h >> 8u) + 0.5) / 16777216.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(dst);
  if any(id.xy >= dims) {
    return;
  }
  let a = pcg((id.y * dims.x + id.x) ^ pcg(u32(p.seed)));
  let b = pcg(a);
  let z = sqrt(-2.0 * log(unit(a))) * cos(6.2831853 * unit(b));
  textureStore(dst, id.xy, vec4f(textureLoad(src, id.xy, 0).rgb, z));
}
