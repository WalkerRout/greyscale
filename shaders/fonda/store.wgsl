@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var original: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let c = floor(clamp(textureLoad(src, id.xy, 0).rgb, vec3f(0.0), vec3f(1.0)) * 255.0 + 0.5) / 255.0;
  textureStore(dst, id.xy, vec4f(c, textureLoad(original, id.xy, 0).a));
}
