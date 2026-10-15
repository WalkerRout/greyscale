@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec2f>;

var<workgroup> scratch: array<vec2f, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_index) local: u32,
  @builtin(workgroup_id) group: vec3u,
  @builtin(num_workgroups) groups: vec3u,
) {
  var v = vec2f(0.0);
  if all(id.xy < textureDimensions(src)) {
    let x = textureLoad(src, id.xy, 0).a;
    v = vec2f(x, x * x);
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
    partials[group.y * groups.x + group.x] = scratch[0];
  }
}
