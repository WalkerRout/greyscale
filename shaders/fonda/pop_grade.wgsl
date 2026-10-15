@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { exposure: f32, chroma: f32, lift: f32 }
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let lab = rgb_to_oklab(textureLoad(src, id.xy, 0).rgb);
  var l = lab.x + p.exposure * 4.0 * lab.x * (1.0 - lab.x);
  l = p.lift + (1.0 - p.lift) * l;
  textureStore(dst, id.xy, vec4f(oklab_to_rgb(vec3f(clamp(l, 0.0, 1.0), lab.yz * p.chroma)), 1.0));
}
