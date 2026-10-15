fn smooth_step(e0: f32, e1: f32, x: f32) -> f32 {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn soft_tanh(x: f32) -> f32 {
  return tanh(clamp(x, -15.0, 15.0));
}

fn cbrt(v: vec3f) -> vec3f {
  return select(vec3f(0.0), pow(max(v, vec3f(1e-30)), vec3f(1.0 / 3.0)), v > vec3f(0.0));
}

fn to_linear(c: vec3f) -> vec3f {
  return select(pow((max(c, vec3f(0.0)) + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn to_srgb(c: vec3f) -> vec3f {
  let x = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(max(x, vec3f(1e-30)), vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

fn rgb_to_oklab(rgb: vec3f) -> vec3f {
  let lms = to_linear(rgb) * mat3x3f(
    vec3f(0.4122214708, 0.5363325363, 0.0514459929),
    vec3f(0.2119034982, 0.6806995451, 0.1073969566),
    vec3f(0.0883024619, 0.2817188376, 0.6299787005),
  );
  return cbrt(max(lms, vec3f(0.0))) * mat3x3f(
    vec3f(0.2104542553, 0.7936177850, -0.0040720468),
    vec3f(1.9779984951, -2.4285922050, 0.4505937099),
    vec3f(0.0259040371, 0.7827717662, -0.8086757660),
  );
}

fn oklab_to_rgb(lab: vec3f) -> vec3f {
  let l = lab * mat3x3f(
    vec3f(1.0, 0.3963377774, 0.2158037573),
    vec3f(1.0, -0.1055613458, -0.0638541728),
    vec3f(1.0, -0.0894841775, -1.2914855480),
  );
  return to_srgb((l * l * l) * mat3x3f(
    vec3f(4.0767416621, -3.3077115913, 0.2309699292),
    vec3f(-1.2684380046, 2.6097574011, -0.3413193965),
    vec3f(-0.0041960863, -0.7034186147, 1.7076147010),
  ));
}

fn lab_f(t: vec3f) -> vec3f {
  return select(7.787 * t + 16.0 / 116.0, cbrt(t), t > vec3f(0.008856));
}

fn rgb_to_cvlab(rgb: vec3f) -> vec3f {
  let xyz = to_linear(rgb) * mat3x3f(
    vec3f(0.412453, 0.357580, 0.180423) / 0.950456,
    vec3f(0.212671, 0.715160, 0.072169),
    vec3f(0.019334, 0.119193, 0.950227) / 1.088754,
  );
  let f = lab_f(xyz);
  let l = select(903.3 * xyz.y, 116.0 * f.y - 16.0, xyz.y > 0.008856);
  return vec3f(l, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z));
}

fn cvlab_to_rgb(lab: vec3f) -> vec3f {
  var y = lab.x / 903.3;
  var fy = 7.787 * y + 16.0 / 116.0;
  if lab.x > 0.008856 * 903.3 {
    fy = (lab.x + 16.0) / 116.0;
    y = fy * fy * fy;
  }
  let fxz = vec2f(lab.y / 500.0 + fy, fy - lab.z / 200.0);
  let xz = select(fxz * fxz * fxz, (fxz - 16.0 / 116.0) / 7.787, fxz <= vec2f(7.787 * 0.008856 + 16.0 / 116.0));
  return to_srgb(vec3f(xz.x * 0.950456, y, xz.y * 1.088754) * mat3x3f(
    vec3f(3.240479, -1.53715, -0.498535),
    vec3f(-0.969256, 1.875991, 0.041556),
    vec3f(0.055648, -0.204043, 1.057311),
  ));
}

fn long_side(t: texture_2d<f32>) -> f32 {
  let n = textureDimensions(t);
  return f32(max(n.x, n.y));
}

fn reflect_index(i: i32, n: i32) -> i32 {
  let period = 2 * n;
  let j = ((i % period) + period) % period;
  return select(j, period - 1 - j, j >= n);
}

fn reflect101_index(i: i32, n: i32) -> i32 {
  if n == 1 {
    return 0;
  }
  let period = 2 * n - 2;
  let j = ((i % period) + period) % period;
  return select(j, period - j, j >= n);
}

fn fetch(t: texture_2d<f32>, q: vec2i, mirror: bool) -> vec4f {
  let n = vec2i(textureDimensions(t));
  var r = clamp(q, vec2i(0), n - 1);
  if mirror {
    r = vec2i(reflect_index(q.x, n.x), reflect_index(q.y, n.y));
  }
  return textureLoad(t, r, 0);
}

fn bilinear(t: texture_2d<f32>, p: vec2f, mirror: bool) -> vec4f {
  var q = p;
  if !mirror {
    q = clamp(p, vec2f(0.0), vec2f(textureDimensions(t)) - 1.0);
  }
  let base = floor(q);
  let f = q - base;
  let i = vec2i(base);
  let top = mix(fetch(t, i, mirror), fetch(t, i + vec2i(1, 0), mirror), f.x);
  let bottom = mix(fetch(t, i + vec2i(0, 1), mirror), fetch(t, i + vec2i(1, 1), mirror), f.x);
  return mix(top, bottom, f.y);
}
