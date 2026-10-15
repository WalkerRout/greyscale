@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var aux1: texture_2d<f32>;
@group(0) @binding(3) var aux2: texture_2d<f32>;
struct Params { lo: f32, hi: f32 }
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn luma(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(rgb_to_cvlab(textureLoad(src, id.xy, 0).rgb).x / 100.0));
}

@compute @workgroup_size(16, 16)
fn detail(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let l = textureLoad(src, id.xy, 0);
  textureStore(dst, id.xy, vec4f(abs(l.a - l.r)));
}

@compute @workgroup_size(16, 16)
fn area(@builtin(global_invocation_id) id: vec3u) {
  let extent = textureDimensions(dst);
  if any(id.xy >= extent) {
    return;
  }
  let n = vec2f(textureDimensions(src));
  let scale = n / vec2f(extent);
  let lo = vec2f(id.xy) * scale;
  let hi = min(lo + scale, n);
  var sum = vec4f(0.0);
  var total = 0.0;
  for (var y = i32(floor(lo.y)); y < i32(ceil(hi.y)); y++) {
    let wy = min(f32(y + 1), hi.y) - max(f32(y), lo.y);
    for (var x = i32(floor(lo.x)); x < i32(ceil(hi.x)); x++) {
      let w = wy * (min(f32(x + 1), hi.x) - max(f32(x), lo.x));
      sum += w * textureLoad(src, vec2i(x, y), 0);
      total += w;
    }
  }
  textureStore(dst, id.xy, sum / total);
}

@compute @workgroup_size(16, 16)
fn blend(@builtin(global_invocation_id) id: vec3u) {
  let full = textureDimensions(src);
  if any(id.xy >= full) {
    return;
  }
  let at = (vec2f(id.xy) + 0.5) * vec2f(textureDimensions(aux2)) / vec2f(full) - 0.5;
  let blob = bilinear(aux2, at, false).rgb;
  let m = smooth_step(p.lo, p.hi, textureLoad(aux1, id.xy, 0).r);
  textureStore(dst, id.xy, vec4f(textureLoad(src, id.xy, 0).rgb * (1.0 - m) + blob * m, 1.0));
}
