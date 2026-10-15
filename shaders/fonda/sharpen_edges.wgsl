@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var aux: texture_2d<f32>;
struct Params { amount: f32, threshold: f32 }
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn lightness(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(rgb_to_oklab(textureLoad(src, id.xy, 0).rgb).x));
}

@compute @workgroup_size(16, 16)
fn apply(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let lab = rgb_to_oklab(textureLoad(src, id.xy, 0).rgb);
  let d = lab.x - textureLoad(aux, id.xy, 0).r;
  let shaped = sign(d) * max(abs(d) - p.threshold, 0.0);
  textureStore(dst, id.xy, vec4f(oklab_to_rgb(vec3f(clamp(lab.x + p.amount * shaped, 0.0, 1.0), lab.yz)), 1.0));
}
