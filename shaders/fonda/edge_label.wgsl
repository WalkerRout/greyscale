@group(0) @binding(0) var<storage, read_write> labels: array<atomic<u32>>;
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> sizes: array<atomic<u32>>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;
struct Params { min_len: f32 }
@group(0) @binding(4) var<uniform> p: Params;

fn foreground(q: vec2i) -> bool {
  return textureLoad(mask, q, 0).r > 0.3;
}

fn index(q: vec2i) -> u32 {
  return u32(q.y) * textureDimensions(mask).x + u32(q.x);
}

fn find(i: u32) -> u32 {
  var x = i;
  var parent = atomicLoad(&labels[x]);
  while parent != x {
    x = parent;
    parent = atomicLoad(&labels[x]);
  }
  return x;
}

fn unite(i: u32, j: u32) {
  var a = find(i);
  var b = find(j);
  while a != b {
    if a > b {
      let t = a;
      a = b;
      b = t;
    }
    let old = atomicMin(&labels[b], a);
    if old == b {
      break;
    }
    a = find(a);
    b = find(old);
  }
}

fn kept(q: vec2i) -> bool {
  if any(q < vec2i(0)) || any(q >= vec2i(textureDimensions(mask))) || !foreground(q) {
    return false;
  }
  return f32(atomicLoad(&sizes[atomicLoad(&labels[index(q)])])) >= p.min_len * long_side(mask) / 1600.0;
}

@compute @workgroup_size(16, 16)
fn init(@builtin(global_invocation_id) id: vec3u) {
  if any(id.xy >= textureDimensions(mask)) {
    return;
  }
  let i = index(vec2i(id.xy));
  atomicStore(&labels[i], i);
}

@compute @workgroup_size(16, 16)
fn merge(@builtin(global_invocation_id) id: vec3u) {
  let q = vec2i(id.xy);
  if any(id.xy >= textureDimensions(mask)) || !foreground(q) {
    return;
  }
  if q.x > 0 && foreground(q - vec2i(1, 0)) {
    unite(index(q), index(q - vec2i(1, 0)));
  }
  if q.y > 0 && foreground(q - vec2i(0, 1)) {
    unite(index(q), index(q - vec2i(0, 1)));
  }
}

@compute @workgroup_size(16, 16)
fn count(@builtin(global_invocation_id) id: vec3u) {
  let q = vec2i(id.xy);
  if any(id.xy >= textureDimensions(mask)) {
    return;
  }
  let i = index(q);
  let root = find(i);
  atomicStore(&labels[i], root);
  if foreground(q) {
    atomicAdd(&sizes[root], 1u);
    atomicAdd(&sizes[arrayLength(&sizes) - 1u], 1u);
  }
}

fn pruned(q: vec2i) -> f32 {
  var keep = atomicLoad(&sizes[arrayLength(&sizes) - 1u]) == 0u;
  keep = keep || kept(q) || kept(q + vec2i(1, 0)) || kept(q - vec2i(1, 0));
  keep = keep || kept(q + vec2i(0, 1)) || kept(q - vec2i(0, 1));
  return textureLoad(mask, q, 0).r * select(0.0, 1.0, keep);
}

@compute @workgroup_size(16, 16)
fn prune(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(mask));
  if any(vec2i(id.xy) >= dims) {
    return;
  }
  var v = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      v = max(v, pruned(clamp(vec2i(id.xy) + vec2i(dx, dy), vec2i(0), dims - 1)));
    }
  }
  textureStore(dst, id.xy, vec4f(v));
}
