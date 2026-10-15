@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var flow: texture_2d<f32>;
struct Params { length: f32, steps: f32 }
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let origin = vec2f(id.xy);
  let t = textureLoad(flow, id.xy, 0).xy;
  var acc = textureLoad(src, id.xy, 0);
  let reach = p.length * long_side(src) / 1600.0;
  var n = 1.0;
  for (var k = 1; k <= i32(p.steps); k++) {
    let d = reach * f32(k) / p.steps;
    acc += bilinear(src, origin + d * t, false) + bilinear(src, origin - d * t, false);
    n += 2.0;
  }
  textureStore(dst, id.xy, acc / n);
}
