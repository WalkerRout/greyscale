@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var source: texture_2d<f32>;
@group(0) @binding(3) var edges: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let e = textureLoad(edges, id.xy, 0).r;
  let c = textureLoad(src, id.xy, 0).rgb * (1.0 - e) + textureLoad(source, id.xy, 0).rgb * e;
  textureStore(dst, id.xy, vec4f(c, 1.0));
}
