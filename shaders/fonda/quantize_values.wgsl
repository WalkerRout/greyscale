@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { bins: f32, phi_lo: f32, phi_hi: f32 }
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let centre = vec2i(id.xy);
  var gradient = vec2f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let l = rgb_to_cvlab(fetch(src, centre + vec2i(dx, dy), true).rgb).x;
      gradient += l * vec2f(f32(dx * (2 - dy * dy)), f32(dy * (2 - dx * dx)));
    }
  }
  let lab = rgb_to_cvlab(textureLoad(src, centre, 0).rgb);
  let dq = 100.0 / p.bins;
  let q = round(lab.x / dq) * dq;
  let phi = p.phi_lo + (p.phi_hi - p.phi_lo) * clamp(length(gradient) / 8.0 / 2.0, 0.0, 1.0);
  let l = q + dq / 2.0 * soft_tanh(phi * (lab.x - q) / 10.0);
  textureStore(dst, id.xy, vec4f(cvlab_to_rgb(vec3f(l, lab.yz)), 1.0));
}
