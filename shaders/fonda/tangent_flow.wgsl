@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn structure(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  var g = vec2f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let l = rgb_to_cvlab(fetch(src, vec2i(id.xy) + vec2i(dx, dy), true).rgb).x / 100.0;
      g += l * vec2f(f32(dx * (2 - dy * dy)), f32(dy * (2 - dx * dx)));
    }
  }
  textureStore(dst, id.xy, vec4f(g.x * g.x, g.x * g.y, g.y * g.y, 0.0));
}

@compute @workgroup_size(16, 16)
fn eigen(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let t = textureLoad(src, id.xy, 0);
  let root = sqrt((t.x - t.z) * (t.x - t.z) + 4.0 * t.y * t.y);
  let major = 0.5 * (t.x + t.z + root);
  let minor = 0.5 * (t.x + t.z - root);
  var tangent = vec2f(major - t.x, -t.y);
  let n = length(tangent) + 1e-8;
  tangent = select(tangent / n, vec2f(1.0, 0.0), n < 1e-6);
  let c = (major - minor) / (major + minor + 1e-4);
  textureStore(dst, id.xy, vec4f(tangent, c * c, 0.0));
}
