@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params {
  shadow_a: f32,
  shadow_b: f32,
  lift: f32,
  chroma_boost: f32,
  light_a: f32,
  light_b: f32,
  light_amount: f32,
}
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let lab = rgb_to_oklab(textureLoad(src, id.xy, 0).rgb);
  let w = 1.0 - smooth_step(0.30, 0.70, lab.x);
  var ab = lab.yz * (1.0 + p.chroma_boost * w) + vec2f(p.shadow_a, p.shadow_b) * w;
  let l = lab.x + p.lift * w * (1.0 - lab.x);
  ab += vec2f(p.light_a, p.light_b) * smooth_step(0.65, 0.95, l) * p.light_amount;
  textureStore(dst, id.xy, vec4f(oklab_to_rgb(vec3f(l, ab)), 1.0));
}
