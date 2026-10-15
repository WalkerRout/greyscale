@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { contrast_lo: f32, contrast_hi: f32, busy_lo: f32, busy_hi: f32 }
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn luma(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(rgb_to_cvlab(textureLoad(src, id.xy, 0).rgb).x / 100.0));
}

@compute @workgroup_size(16, 16)
fn line(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let l = textureLoad(src, id.xy, 0);
  let m = smooth_step(p.contrast_lo, p.contrast_hi, abs(l.r - l.g));
  textureStore(dst, id.xy, vec4f(m, abs(l.a - l.b), 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn gate(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let line = textureLoad(src, id.xy, 0);
  textureStore(dst, id.xy, vec4f(line.r * (1.0 - smooth_step(p.busy_lo, p.busy_hi, line.g))));
}
