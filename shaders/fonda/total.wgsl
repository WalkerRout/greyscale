@group(0) @binding(0) var<storage, read> partials: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> deviation: array<f32>;
@group(0) @binding(2) var values: texture_2d<f32>;

var<workgroup> scratch: array<vec2f, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32) {
  var v = vec2f(0.0);
  for (var i = local; i < arrayLength(&partials); i += 256u) {
    v += partials[i];
  }
  scratch[local] = v;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if local < stride {
      scratch[local] += scratch[local + stride];
    }
    workgroupBarrier();
  }
  if local == 0u {
    let n = textureDimensions(values);
    let count = f32(n.x * n.y);
    let mean = scratch[0].x / count;
    deviation[0] = sqrt(max(scratch[0].y / count - mean * mean, 0.0));
  }
}
