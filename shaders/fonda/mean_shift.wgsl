@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
struct Params { spatial: f32, colour: f32 }
@group(0) @binding(2) var<uniform> p: Params;

fn texel(q: vec2i) -> vec3f {
  return floor(clamp(textureLoad(src, q, 0).rgb, vec3f(0.0), vec3f(1.0)) * 255.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(src));
  var pos = vec2i(id.xy);
  if any(pos >= dims) {
    return;
  }
  var colour = texel(pos);
  let spatial = max(4.0, floor(long_side(src) * p.spatial));
  let radius2 = p.colour * p.colour;
  for (var iter = 0; iter < 5; iter++) {
    let lo = max(vec2i(round(vec2f(pos) - spatial)), vec2i(0));
    let hi = min(vec2i(round(vec2f(pos) + spatial)), dims - 1);
    var sum_colour = vec3f(0.0);
    var sum_pos = vec2f(0.0);
    var count = 0.0;
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let t = texel(vec2i(x, y));
        let d = t - colour;
        if dot(d, d) <= radius2 {
          sum_colour += t;
          sum_pos += vec2f(f32(x), f32(y));
          count += 1.0;
        }
      }
    }
    if count == 0.0 {
      break;
    }
    let next_pos = vec2i(round(sum_pos / count));
    let next_colour = round(sum_colour / count);
    let shift = abs(next_pos - pos);
    let dc = next_colour - colour;
    let converged = all(next_pos == pos) || f32(shift.x + shift.y) + dot(dc, dc) <= 1.0;
    pos = next_pos;
    colour = next_colour;
    if converged {
      break;
    }
  }
  textureStore(dst, id.xy, vec4f(colour / 255.0, 1.0));
}
