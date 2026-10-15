@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var flow: texture_2d<f32>;
struct Params { length: f32, steps: f32, amount: f32 }
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read> deviation: array<f32>;
@group(0) @binding(5) var strokes: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn lic(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let step_size = p.length * long_side(src) / p.steps;
  var acc = textureLoad(src, id.xy, 0).a;
  for (var side = 0; side < 2; side++) {
    let dir = select(-1.0, 1.0, side == 0);
    var at = vec2f(id.xy);
    for (var i = 0; i < i32(p.steps); i++) {
      at += dir * step_size * bilinear(flow, at, false).xy;
      acc += bilinear(src, at, true).a;
    }
  }
  textureStore(dst, id.xy, vec4f(acc / (2.0 * p.steps + 1.0)));
}

@compute @workgroup_size(16, 16)
fn apply(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let lab = rgb_to_oklab(textureLoad(src, id.xy, 0).rgb);
  let stroke = textureLoad(strokes, id.xy, 0).r / (deviation[0] + 1e-6);
  let coherence = textureLoad(flow, id.xy, 0).z;
  let l = lab.x + p.amount * stroke * coherence * (0.3 + 0.7 * 4.0 * lab.x * (1.0 - lab.x));
  textureStore(dst, id.xy, vec4f(oklab_to_rgb(vec3f(clamp(l, 0.0, 1.0), lab.yz)), 1.0));
}
