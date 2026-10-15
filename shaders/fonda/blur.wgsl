@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { sigma: vec4f, factor: vec4f, reference: vec4f }
@group(0) @binding(2) var<uniform> p: Params;

fn sigma() -> vec4f {
  let s = p.sigma * p.factor;
  return select(s, s * long_side(src) / p.reference, p.reference > vec4f(0.0));
}

fn convolve(at: vec2i, dir: vec2i) -> vec4f {
  let s = sigma();
  let enabled = s > vec4f(0.0);
  let radius = select(vec4i(0), vec4i(4.0 * s + 0.5), enabled);
  let reach = max(max(radius.x, radius.y), max(radius.z, radius.w));
  var sum = vec4f(0.0);
  var total = vec4f(0.0);
  for (var k = -reach; k <= reach; k++) {
    let gauss = select(vec4f(select(0.0, 1.0, k == 0)), exp(-0.5 * f32(k * k) / (s * s)), enabled);
    let w = select(vec4f(0.0), gauss, vec4i(abs(k)) <= radius);
    sum += w * fetch(src, at + k * dir, true);
    total += w;
  }
  return sum / total;
}

@compute @workgroup_size(16, 16)
fn horizontal(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, convolve(vec2i(id.xy), vec2i(1, 0)));
}

@compute @workgroup_size(16, 16)
fn vertical(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, convolve(vec2i(id.xy), vec2i(0, 1)));
}
