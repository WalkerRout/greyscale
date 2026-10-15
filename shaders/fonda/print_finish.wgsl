@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { red: f32, green: f32, blue: f32, ink: f32, grain: f32 }
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> deviation: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  let texel = textureLoad(src, id.xy, 0);
  let base = max(1.0 - texel.rgb, vec3f(0.0));
  let inked = 1.0 - select(vec3f(0.0), pow(max(base, vec3f(1e-30)), vec3f(p.ink)), base > vec3f(0.0));
  let grain = 1.0 + p.grain * texel.a / (deviation[0] + 1e-6);
  let c = clamp(inked * vec3f(p.red, p.green, p.blue) * grain, vec3f(0.0), vec3f(1.0));
  textureStore(dst, id.xy, vec4f(c, 1.0));
}
