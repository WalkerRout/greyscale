@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { sigma_d: f32, sigma_r: f32 }
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn to_lab(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(rgb_to_cvlab(textureLoad(src, id.xy, 0).rgb), 1.0));
}

fn filter_lab(centre: vec2i) -> vec3f {
  let dims = vec2i(textureDimensions(src));
  let c0 = textureLoad(src, centre, 0).rgb;
  let sigma_space = p.sigma_d * long_side(src) / 512.0;
  let sigma_colour = p.sigma_r * 2.0;
  let r = i32(round(2.0 * sigma_space));
  var sum = vec3f(0.0);
  var total = 0.0;
  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let d2 = dx * dx + dy * dy;
      if d2 > r * r {
        continue;
      }
      let q = vec2i(reflect101_index(centre.x + dx, dims.x), reflect101_index(centre.y + dy, dims.y));
      let c = textureLoad(src, q, 0).rgb;
      let diff = abs(c - c0);
      let dist = diff.x + diff.y + diff.z;
      let w = exp(-0.5 * f32(d2) / (sigma_space * sigma_space) - 0.5 * dist * dist / (sigma_colour * sigma_colour));
      sum += w * c;
      total += w;
    }
  }
  return sum / total;
}

@compute @workgroup_size(16, 16)
fn bilateral(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(filter_lab(vec2i(id.xy)), 1.0));
}

@compute @workgroup_size(16, 16)
fn bilateral_to_rgb(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(cvlab_to_rgb(filter_lab(vec2i(id.xy))), 1.0));
}

@compute @workgroup_size(16, 16)
fn from_lab(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(src)) {
    return;
  }
  textureStore(dst, id.xy, vec4f(cvlab_to_rgb(textureLoad(src, id.xy, 0).rgb), 1.0));
}
