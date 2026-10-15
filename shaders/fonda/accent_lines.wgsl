@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { tau: f32, phi: f32, opacity: f32, red: f32, green: f32, blue: f32 }
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var aux: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn luma(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(rgb_to_cvlab(textureLoad(src, id.xy, 0).rgb).x));
}

@compute @workgroup_size(16, 16)
fn line(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let blurred = textureLoad(src, id.xy, 0);
  let d = blurred.r - p.tau * blurred.g;
  textureStore(dst, id.xy, vec4f(clamp(-soft_tanh(p.phi * d / 4.0), 0.0, 1.0)));
}

@compute @workgroup_size(16, 16)
fn compose(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let a = textureLoad(aux, id.xy, 0).r * p.opacity;
  let c = textureLoad(src, id.xy, 0).rgb * (1.0 - a) + vec3f(p.red, p.green, p.blue) * a;
  textureStore(dst, id.xy, vec4f(c, 1.0));
}
