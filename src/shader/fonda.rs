use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
  BindingResource, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline,
  Extent3d, Texture, TextureFormat, TextureView, TextureViewDescriptor,
};

use crate::gpu::{self, GpuContext, Scratch};
use crate::shader::{
  Kernel, WORKGROUP_SIZE, create_bind_group, create_compute_pipeline, create_shader_module,
  dispatch,
};

pub const FORMAT: TextureFormat = TextureFormat::Rgba32Float;

const COMMON: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/common.wgsl"
));

const BLUR: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/blur.wgsl"
));

const PARTIAL: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/partial.wgsl"
));

const TOTAL: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/total.wgsl"
));

const NOISE: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/noise.wgsl"
));

const MEAN_SHIFT: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/mean_shift.wgsl"
));

const POP_GRADE: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/pop_grade.wgsl"
));

const ABSTRACT: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/abstract.wgsl"
));

const EDGE_MAP: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/edge_map.wgsl"
));

const EDGE_LABEL: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/edge_label.wgsl"
));

const SIMPLIFY_TEXTURE: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/simplify_texture.wgsl"
));

const PROTECT_EDGES: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/protect_edges.wgsl"
));

const QUANTIZE_VALUES: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/quantize_values.wgsl"
));

const COLOUR_SHADOWS: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/colour_shadows.wgsl"
));

const ACCENT_LINES: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/accent_lines.wgsl"
));

const TANGENT_FLOW: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/tangent_flow.wgsl"
));

const BRUSH_TEXTURE: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/brush_texture.wgsl"
));

const FLOW_ANTIALIAS: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/flow_antialias.wgsl"
));

const SHARPEN_EDGES: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/sharpen_edges.wgsl"
));

const PRINT_FINISH: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/print_finish.wgsl"
));

const STORE: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/shaders/fonda/store.wgsl"
));

enum Bind<'a> {
  Tex(&'a Texture),
  Buf(&'a Buffer),
}

type Pair = (ComputePipeline, ComputePipeline);

fn pipelines<const N: usize>(
  ctx: &GpuContext,
  label: &str,
  source: &str,
  entries: [&str; N],
) -> [ComputePipeline; N] {
  let module = create_shader_module(ctx, label, &format!("{COMMON}{source}"));
  entries.map(|entry| create_compute_pipeline(ctx, label, &module, entry))
}

fn pipeline(ctx: &GpuContext, label: &str, source: &str) -> ComputePipeline {
  let [main] = pipelines(ctx, label, source, ["main"]);
  main
}

fn blur_pipelines(ctx: &GpuContext) -> Pair {
  let [horizontal, vertical] = pipelines(ctx, "fonda blur", BLUR, ["horizontal", "vertical"]);
  (horizontal, vertical)
}

fn deviation_pipelines(ctx: &GpuContext) -> Pair {
  (
    pipeline(ctx, "fonda partial sums", PARTIAL),
    pipeline(ctx, "fonda deviation", TOTAL),
  )
}

fn dims(texture: &Texture) -> (u32, u32) {
  (texture.width(), texture.height())
}

fn scratch(ctx: &GpuContext, texture: &Texture) -> Scratch {
  ctx.scratch(texture.size(), FORMAT)
}

fn uniform(ctx: &GpuContext, values: &[f32]) -> Buffer {
  let mut data = values.to_vec();
  data.resize(values.len().div_ceil(4).max(1) * 4, 0.0);
  ctx.device.create_buffer_init(&BufferInitDescriptor {
    label: Some("fonda uniform"),
    contents: bytemuck::cast_slice(&data),
    usage: BufferUsages::UNIFORM,
  })
}

fn zeroed(ctx: &GpuContext, size: u64) -> Buffer {
  ctx.device.create_buffer(&BufferDescriptor {
    label: Some("fonda storage"),
    size,
    usage: BufferUsages::STORAGE,
    mapped_at_creation: false,
  })
}

fn run(
  ctx: &GpuContext,
  encoder: &mut CommandEncoder,
  pipeline: &ComputePipeline,
  grid: (u32, u32),
  bindings: &[Bind],
) {
  let views: Vec<TextureView> = bindings
    .iter()
    .filter_map(|binding| match binding {
      Bind::Tex(texture) => Some(texture.create_view(&TextureViewDescriptor::default())),
      Bind::Buf(_) => None,
    })
    .collect();
  let mut views = views.iter();
  let resources: Vec<BindingResource> = bindings
    .iter()
    .map(|binding| match binding {
      Bind::Tex(_) => BindingResource::TextureView(views.next().unwrap()),
      Bind::Buf(buffer) => buffer.as_entire_binding(),
    })
    .collect();
  let bind_group = create_bind_group(ctx, "fonda bind group", pipeline, &resources);
  dispatch(encoder, pipeline, &bind_group, grid);
}

fn gaussian(
  ctx: &GpuContext,
  encoder: &mut CommandEncoder,
  (horizontal, vertical): &Pair,
  input: &Texture,
  output: &Texture,
  [sigma, factor, reference]: [[f32; 4]; 3],
) {
  let params = uniform(ctx, &[sigma, factor, reference].concat());
  let tmp = scratch(ctx, input);
  run(
    ctx,
    encoder,
    horizontal,
    dims(input),
    &[Bind::Tex(input), Bind::Tex(&tmp), Bind::Buf(&params)],
  );
  run(
    ctx,
    encoder,
    vertical,
    dims(input),
    &[Bind::Tex(&tmp), Bind::Tex(output), Bind::Buf(&params)],
  );
}

fn deviation(
  ctx: &GpuContext,
  encoder: &mut CommandEncoder,
  (partial, total): &Pair,
  input: &Texture,
) -> Buffer {
  let (x, y) = gpu::compute_work_group_count(dims(input), WORKGROUP_SIZE);
  let partials = zeroed(ctx, (x * y) as u64 * 8);
  let result = zeroed(ctx, 4);
  run(
    ctx,
    encoder,
    partial,
    dims(input),
    &[Bind::Tex(input), Bind::Buf(&partials)],
  );
  run(
    ctx,
    encoder,
    total,
    (1, 1),
    &[Bind::Buf(&partials), Bind::Buf(&result), Bind::Tex(input)],
  );
  result
}

pub struct PopGrade {
  pub exposure: f32,
  pub chroma: f32,
  pub lift: f32,
  context: GpuContext,
  pipeline: ComputePipeline,
}

impl PopGrade {
  pub fn new(context: &GpuContext) -> Self {
    Self {
      exposure: 0.09,
      chroma: 1.35,
      lift: 0.04,
      context: context.clone(),
      pipeline: pipeline(context, "fonda pop grade", POP_GRADE),
    }
  }
}

impl Kernel for PopGrade {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let params = uniform(&self.context, &[self.exposure, self.chroma, self.lift]);
    run(
      &self.context,
      encoder,
      &self.pipeline,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(output), Bind::Buf(&params)],
    );
  }
}

pub struct Abstract {
  pub iterations: u32,
  pub sigma_d: f32,
  pub sigma_r: f32,
  context: GpuContext,
  to_lab: ComputePipeline,
  bilateral: ComputePipeline,
  bilateral_to_rgb: ComputePipeline,
  from_lab: ComputePipeline,
}

impl Abstract {
  pub fn new(context: &GpuContext) -> Self {
    let [to_lab, bilateral, bilateral_to_rgb, from_lab] = pipelines(
      context,
      "fonda abstract",
      ABSTRACT,
      ["to_lab", "bilateral", "bilateral_to_rgb", "from_lab"],
    );
    Self {
      iterations: 4,
      sigma_d: 3.0,
      sigma_r: 4.25,
      context: context.clone(),
      to_lab,
      bilateral,
      bilateral_to_rgb,
      from_lab,
    }
  }
}

impl Kernel for Abstract {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let ctx = &self.context;
    let params = uniform(ctx, &[self.sigma_d, self.sigma_r]);

    let mut lab = scratch(ctx, input);
    let mut next = scratch(ctx, input);
    run(
      ctx,
      encoder,
      &self.to_lab,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(&lab)],
    );
    if self.iterations == 0 {
      run(
        ctx,
        encoder,
        &self.from_lab,
        dims(input),
        &[Bind::Tex(&lab), Bind::Tex(output)],
      );
      return;
    }
    for _ in 1..self.iterations {
      run(
        ctx,
        encoder,
        &self.bilateral,
        dims(input),
        &[Bind::Tex(&lab), Bind::Tex(&next), Bind::Buf(&params)],
      );
      std::mem::swap(&mut lab, &mut next);
    }
    run(
      ctx,
      encoder,
      &self.bilateral_to_rgb,
      dims(input),
      &[Bind::Tex(&lab), Bind::Tex(output), Bind::Buf(&params)],
    );
  }
}

pub struct FlattenColour {
  pub spatial: f32,
  pub colour: f32,
  context: GpuContext,
  mean_shift: ComputePipeline,
}

impl FlattenColour {
  pub fn new(context: &GpuContext) -> Self {
    Self {
      spatial: 0.012,
      colour: 22.0,
      context: context.clone(),
      mean_shift: pipeline(context, "fonda flatten colour", MEAN_SHIFT),
    }
  }
}

impl Kernel for FlattenColour {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let params = uniform(&self.context, &[self.spatial, self.colour]);
    run(
      &self.context,
      encoder,
      &self.mean_shift,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(output), Bind::Buf(&params)],
    );
  }
}

pub struct SimplifyTexture {
  pub scale: f32,
  pub spatial: f32,
  pub colour: f32,
  pub lo: f32,
  pub hi: f32,
  context: GpuContext,
  blur: Pair,
  luma: ComputePipeline,
  detail: ComputePipeline,
  area: ComputePipeline,
  blend: ComputePipeline,
  mean_shift: ComputePipeline,
}

impl SimplifyTexture {
  pub fn new(context: &GpuContext) -> Self {
    let [luma, detail, area, blend] = pipelines(
      context,
      "fonda simplify texture",
      SIMPLIFY_TEXTURE,
      ["luma", "detail", "area", "blend"],
    );
    Self {
      scale: 0.5,
      spatial: 0.02,
      colour: 34.0,
      lo: 0.035,
      hi: 0.08,
      context: context.clone(),
      blur: blur_pipelines(context),
      luma,
      detail,
      area,
      blend,
      mean_shift: pipeline(context, "fonda simplify texture mean shift", MEAN_SHIFT),
    }
  }
}

impl Kernel for SimplifyTexture {
  type Input<'a> = (&'a Texture, &'a Texture);

  fn encode(
    &self,
    encoder: &mut CommandEncoder,
    (image, orig): (&Texture, &Texture),
    output: &Texture,
  ) {
    let ctx = &self.context;

    let l = scratch(ctx, image);
    let local = scratch(ctx, image);
    let raw = scratch(ctx, image);
    let detail = scratch(ctx, image);
    run(
      ctx,
      encoder,
      &self.luma,
      dims(image),
      &[Bind::Tex(orig), Bind::Tex(&l)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &l,
      &local,
      [[2.0, 0.0, 0.0, 0.0], [1.0; 4], [1600.0; 4]],
    );
    run(
      ctx,
      encoder,
      &self.detail,
      dims(image),
      &[Bind::Tex(&local), Bind::Tex(&raw)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &raw,
      &detail,
      [[6.0; 4], [1.0; 4], [1600.0; 4]],
    );

    let small_size = Extent3d {
      width: ((image.width() as f32 * self.scale) as u32).max(1),
      height: ((image.height() as f32 * self.scale) as u32).max(1),
      depth_or_array_layers: 1,
    };
    let small = ctx.scratch(small_size, FORMAT);
    let blob = ctx.scratch(small_size, FORMAT);
    run(
      ctx,
      encoder,
      &self.area,
      dims(&small),
      &[Bind::Tex(image), Bind::Tex(&small)],
    );
    let shift = uniform(ctx, &[self.spatial, self.colour]);
    run(
      ctx,
      encoder,
      &self.mean_shift,
      dims(&small),
      &[Bind::Tex(&small), Bind::Tex(&blob), Bind::Buf(&shift)],
    );

    let params = uniform(ctx, &[self.lo, self.hi]);
    run(
      ctx,
      encoder,
      &self.blend,
      dims(image),
      &[
        Bind::Tex(image),
        Bind::Tex(output),
        Bind::Tex(&detail),
        Bind::Tex(&blob),
        Bind::Buf(&params),
      ],
    );
  }
}

pub struct QuantizeValues {
  pub bins: u32,
  pub phi_lo: f32,
  pub phi_hi: f32,
  context: GpuContext,
  pipeline: ComputePipeline,
}

impl QuantizeValues {
  pub fn new(context: &GpuContext) -> Self {
    Self {
      bins: 6,
      phi_lo: 3.0,
      phi_hi: 14.0,
      context: context.clone(),
      pipeline: pipeline(context, "fonda quantize values", QUANTIZE_VALUES),
    }
  }
}

impl Kernel for QuantizeValues {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let params = uniform(&self.context, &[self.bins as f32, self.phi_lo, self.phi_hi]);
    run(
      &self.context,
      encoder,
      &self.pipeline,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(output), Bind::Buf(&params)],
    );
  }
}

pub struct ColourShadows {
  pub shadow_hue: (f32, f32),
  pub lift: f32,
  pub chroma_boost: f32,
  pub light_hue: (f32, f32),
  pub light_amount: f32,
  context: GpuContext,
  pipeline: ComputePipeline,
}

impl ColourShadows {
  pub fn new(context: &GpuContext) -> Self {
    Self {
      shadow_hue: (0.015, -0.045),
      lift: 0.10,
      chroma_boost: 0.6,
      light_hue: (0.004, 0.018),
      light_amount: 0.5,
      context: context.clone(),
      pipeline: pipeline(context, "fonda colour shadows", COLOUR_SHADOWS),
    }
  }
}

impl Kernel for ColourShadows {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let params = uniform(
      &self.context,
      &[
        self.shadow_hue.0,
        self.shadow_hue.1,
        self.lift,
        self.chroma_boost,
        self.light_hue.0,
        self.light_hue.1,
        self.light_amount,
      ],
    );
    run(
      &self.context,
      encoder,
      &self.pipeline,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(output), Bind::Buf(&params)],
    );
  }
}

pub struct AccentLines {
  pub sigma: f32,
  pub k: f32,
  pub tau: f32,
  pub phi: f32,
  pub colour: [f32; 3],
  pub opacity: f32,
  context: GpuContext,
  blur: Pair,
  luma: ComputePipeline,
  line: ComputePipeline,
  compose: ComputePipeline,
}

impl AccentLines {
  pub fn new(context: &GpuContext) -> Self {
    let [luma, line, compose] = pipelines(
      context,
      "fonda accent lines",
      ACCENT_LINES,
      ["luma", "line", "compose"],
    );
    Self {
      sigma: 0.6,
      k: 1.6,
      tau: 0.98,
      phi: 1.2,
      colour: [0.13, 0.11, 0.24],
      opacity: 0.75,
      context: context.clone(),
      blur: blur_pipelines(context),
      luma,
      line,
      compose,
    }
  }
}

impl Kernel for AccentLines {
  type Input<'a> = (&'a Texture, &'a Texture);

  fn encode(
    &self,
    encoder: &mut CommandEncoder,
    (image, guide): (&Texture, &Texture),
    output: &Texture,
  ) {
    let ctx = &self.context;
    let [red, green, blue] = self.colour;
    let params = uniform(ctx, &[self.tau, self.phi, self.opacity, red, green, blue]);

    let l = scratch(ctx, image);
    let blurred = scratch(ctx, image);
    run(
      ctx,
      encoder,
      &self.luma,
      dims(image),
      &[Bind::Tex(guide), Bind::Tex(&l)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &l,
      &blurred,
      [
        [self.sigma, self.sigma, 0.0, 0.0],
        [1.0, self.k, 1.0, 1.0],
        [512.0; 4],
      ],
    );

    let line = scratch(ctx, image);
    let soft = scratch(ctx, image);
    run(
      ctx,
      encoder,
      &self.line,
      dims(image),
      &[Bind::Tex(&blurred), Bind::Tex(&line), Bind::Buf(&params)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &line,
      &soft,
      [[0.6; 4], [1.0; 4], [0.0; 4]],
    );
    run(
      ctx,
      encoder,
      &self.compose,
      dims(image),
      &[
        Bind::Tex(image),
        Bind::Tex(output),
        Bind::Buf(&params),
        Bind::Tex(&soft),
      ],
    );
  }
}

pub struct TangentFlow {
  pub sigma: f32,
  context: GpuContext,
  blur: Pair,
  structure: ComputePipeline,
  eigen: ComputePipeline,
}

impl TangentFlow {
  pub fn new(context: &GpuContext) -> Self {
    let [structure, eigen] = pipelines(
      context,
      "fonda tangent flow",
      TANGENT_FLOW,
      ["structure", "eigen"],
    );
    Self {
      sigma: 2.0,
      context: context.clone(),
      blur: blur_pipelines(context),
      structure,
      eigen,
    }
  }
}

impl Kernel for TangentFlow {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let ctx = &self.context;
    let tensor = scratch(ctx, input);
    let smoothed = scratch(ctx, input);
    run(
      ctx,
      encoder,
      &self.structure,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(&tensor)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &tensor,
      &smoothed,
      [[self.sigma; 4], [1.0; 4], [512.0; 4]],
    );
    run(
      ctx,
      encoder,
      &self.eigen,
      dims(input),
      &[Bind::Tex(&smoothed), Bind::Tex(output)],
    );
  }
}

pub struct BrushTexture {
  pub length: f32,
  pub steps: u32,
  pub amount: f32,
  pub dab: f32,
  pub seed: u32,
  pub flow: TangentFlow,
  context: GpuContext,
  blur: Pair,
  deviation: Pair,
  noise: ComputePipeline,
  lic: ComputePipeline,
  apply: ComputePipeline,
}

impl BrushTexture {
  pub fn new(context: &GpuContext) -> Self {
    let [lic, apply] = pipelines(
      context,
      "fonda brush texture",
      BRUSH_TEXTURE,
      ["lic", "apply"],
    );
    Self {
      length: 0.008,
      steps: 6,
      amount: 0.015,
      dab: 1.4,
      seed: 5,
      flow: TangentFlow::new(context),
      context: context.clone(),
      blur: blur_pipelines(context),
      deviation: deviation_pipelines(context),
      noise: pipeline(context, "fonda brush noise", NOISE),
      lic,
      apply,
    }
  }
}

impl Kernel for BrushTexture {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let ctx = &self.context;

    let flow = scratch(ctx, input);
    self.flow.encode(encoder, input, &flow);

    let noise = scratch(ctx, input);
    let smooth = scratch(ctx, input);
    let seed = uniform(ctx, &[self.seed as f32]);
    run(
      ctx,
      encoder,
      &self.noise,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(&noise), Bind::Buf(&seed)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &noise,
      &smooth,
      [[0.0, 0.0, 0.0, self.dab], [1.0; 4], [1600.0; 4]],
    );

    let params = uniform(ctx, &[self.length, self.steps as f32, self.amount]);
    let strokes = scratch(ctx, input);
    run(
      ctx,
      encoder,
      &self.lic,
      dims(input),
      &[
        Bind::Tex(&smooth),
        Bind::Tex(&strokes),
        Bind::Tex(&flow),
        Bind::Buf(&params),
      ],
    );
    let std = deviation(ctx, encoder, &self.deviation, &strokes);
    run(
      ctx,
      encoder,
      &self.apply,
      dims(input),
      &[
        Bind::Tex(input),
        Bind::Tex(output),
        Bind::Tex(&flow),
        Bind::Buf(&params),
        Bind::Buf(&std),
        Bind::Tex(&strokes),
      ],
    );
  }
}

pub struct PrintFinish {
  pub paper: [f32; 3],
  pub soften: f32,
  pub ink: f32,
  pub grain: f32,
  pub seed: u32,
  context: GpuContext,
  blur: Pair,
  deviation: Pair,
  noise: ComputePipeline,
  pipeline: ComputePipeline,
}

impl PrintFinish {
  pub fn new(context: &GpuContext) -> Self {
    Self {
      paper: [0.985, 0.965, 0.915],
      soften: 0.3,
      ink: 1.04,
      grain: 0.012,
      seed: 9,
      context: context.clone(),
      blur: blur_pipelines(context),
      deviation: deviation_pipelines(context),
      noise: pipeline(context, "fonda print noise", NOISE),
      pipeline: pipeline(context, "fonda print finish", PRINT_FINISH),
    }
  }
}

impl Kernel for PrintFinish {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let ctx = &self.context;

    let noise = scratch(ctx, input);
    let soft = scratch(ctx, input);
    let seed = uniform(ctx, &[self.seed as f32]);
    run(
      ctx,
      encoder,
      &self.noise,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(&noise), Bind::Buf(&seed)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &noise,
      &soft,
      [
        [self.soften, self.soften, self.soften, 0.8],
        [1.0; 4],
        [1600.0, 1600.0, 1600.0, 0.0],
      ],
    );
    let std = deviation(ctx, encoder, &self.deviation, &soft);

    let [red, green, blue] = self.paper;
    let params = uniform(ctx, &[red, green, blue, self.ink, self.grain]);
    run(
      ctx,
      encoder,
      &self.pipeline,
      dims(input),
      &[
        Bind::Tex(&soft),
        Bind::Tex(output),
        Bind::Buf(&params),
        Bind::Buf(&std),
      ],
    );
  }
}

pub struct EdgeMap {
  pub contrast_lo: f32,
  pub contrast_hi: f32,
  pub busy_lo: f32,
  pub busy_hi: f32,
  pub min_len: f32,
  context: GpuContext,
  blur: Pair,
  luma: ComputePipeline,
  line: ComputePipeline,
  gate: ComputePipeline,
  init: ComputePipeline,
  merge: ComputePipeline,
  count: ComputePipeline,
  prune: ComputePipeline,
}

impl EdgeMap {
  pub fn new(context: &GpuContext) -> Self {
    let [luma, line, gate] = pipelines(
      context,
      "fonda edge map",
      EDGE_MAP,
      ["luma", "line", "gate"],
    );
    let [init, merge, count, prune] = pipelines(
      context,
      "fonda edge labels",
      EDGE_LABEL,
      ["init", "merge", "count", "prune"],
    );
    Self {
      contrast_lo: 0.045,
      contrast_hi: 0.12,
      busy_lo: 0.035,
      busy_hi: 0.08,
      min_len: 20.0,
      context: context.clone(),
      blur: blur_pipelines(context),
      luma,
      line,
      gate,
      init,
      merge,
      count,
      prune,
    }
  }
}

impl Kernel for EdgeMap {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let ctx = &self.context;
    let grid = dims(input);
    let params = uniform(
      ctx,
      &[
        self.contrast_lo,
        self.contrast_hi,
        self.busy_lo,
        self.busy_hi,
      ],
    );

    let l = scratch(ctx, input);
    let scales = scratch(ctx, input);
    run(
      ctx,
      encoder,
      &self.luma,
      grid,
      &[Bind::Tex(input), Bind::Tex(&l)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &l,
      &scales,
      [[0.8, 2.5, 2.0, 0.0], [1.0; 4], [1600.0; 4]],
    );

    let line = scratch(ctx, input);
    let busy = scratch(ctx, input);
    let gated = scratch(ctx, input);
    run(
      ctx,
      encoder,
      &self.line,
      grid,
      &[Bind::Tex(&scales), Bind::Tex(&line), Bind::Buf(&params)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &line,
      &busy,
      [[0.0, 6.0, 0.0, 0.0], [1.0; 4], [1600.0; 4]],
    );
    run(
      ctx,
      encoder,
      &self.gate,
      grid,
      &[Bind::Tex(&busy), Bind::Tex(&gated), Bind::Buf(&params)],
    );

    let pixels = grid.0 as u64 * grid.1 as u64;
    let labels = zeroed(ctx, pixels * 4);
    let sizes = zeroed(ctx, (pixels + 1) * 4);
    let min_len = uniform(ctx, &[self.min_len]);
    run(
      ctx,
      encoder,
      &self.init,
      grid,
      &[Bind::Buf(&labels), Bind::Tex(&gated)],
    );
    run(
      ctx,
      encoder,
      &self.merge,
      grid,
      &[Bind::Buf(&labels), Bind::Tex(&gated)],
    );
    run(
      ctx,
      encoder,
      &self.count,
      grid,
      &[Bind::Buf(&labels), Bind::Tex(&gated), Bind::Buf(&sizes)],
    );

    let dilated = scratch(ctx, input);
    run(
      ctx,
      encoder,
      &self.prune,
      grid,
      &[
        Bind::Buf(&labels),
        Bind::Tex(&gated),
        Bind::Buf(&sizes),
        Bind::Tex(&dilated),
        Bind::Buf(&min_len),
      ],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &dilated,
      output,
      [[0.7; 4], [1.0; 4], [1600.0; 4]],
    );
  }
}

pub struct ProtectEdges {
  context: GpuContext,
  pipeline: ComputePipeline,
}

impl ProtectEdges {
  pub fn new(context: &GpuContext) -> Self {
    Self {
      context: context.clone(),
      pipeline: pipeline(context, "fonda protect edges", PROTECT_EDGES),
    }
  }
}

impl Kernel for ProtectEdges {
  type Input<'a> = (&'a Texture, &'a Texture, &'a Texture);

  fn encode(
    &self,
    encoder: &mut CommandEncoder,
    (image, source, edges): (&Texture, &Texture, &Texture),
    output: &Texture,
  ) {
    run(
      &self.context,
      encoder,
      &self.pipeline,
      dims(image),
      &[
        Bind::Tex(image),
        Bind::Tex(output),
        Bind::Tex(source),
        Bind::Tex(edges),
      ],
    );
  }
}

pub struct FlowAntialias {
  pub length: f32,
  pub steps: u32,
  pub flow: TangentFlow,
  context: GpuContext,
  pipeline: ComputePipeline,
}

impl FlowAntialias {
  pub fn new(context: &GpuContext) -> Self {
    let mut flow = TangentFlow::new(context);
    flow.sigma = 1.0;
    Self {
      length: 1.5,
      steps: 3,
      flow,
      context: context.clone(),
      pipeline: pipeline(context, "fonda flow antialias", FLOW_ANTIALIAS),
    }
  }
}

impl Kernel for FlowAntialias {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let ctx = &self.context;
    let flow = scratch(ctx, input);
    self.flow.encode(encoder, input, &flow);
    let params = uniform(ctx, &[self.length, self.steps as f32]);
    run(
      ctx,
      encoder,
      &self.pipeline,
      dims(input),
      &[
        Bind::Tex(input),
        Bind::Tex(output),
        Bind::Tex(&flow),
        Bind::Buf(&params),
      ],
    );
  }
}

pub struct SharpenEdges {
  pub radius: f32,
  pub amount: f32,
  pub threshold: f32,
  context: GpuContext,
  blur: Pair,
  lightness: ComputePipeline,
  apply: ComputePipeline,
}

impl SharpenEdges {
  pub fn new(context: &GpuContext) -> Self {
    let [lightness, apply] = pipelines(
      context,
      "fonda sharpen edges",
      SHARPEN_EDGES,
      ["lightness", "apply"],
    );
    Self {
      radius: 1.0,
      amount: 1.2,
      threshold: 0.01,
      context: context.clone(),
      blur: blur_pipelines(context),
      lightness,
      apply,
    }
  }
}

impl Kernel for SharpenEdges {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let ctx = &self.context;
    let l = scratch(ctx, input);
    let blurred = scratch(ctx, input);
    run(
      ctx,
      encoder,
      &self.lightness,
      dims(input),
      &[Bind::Tex(input), Bind::Tex(&l)],
    );
    gaussian(
      ctx,
      encoder,
      &self.blur,
      &l,
      &blurred,
      [[self.radius; 4], [1.0; 4], [1600.0; 4]],
    );
    let params = uniform(ctx, &[self.amount, self.threshold]);
    run(
      ctx,
      encoder,
      &self.apply,
      dims(input),
      &[
        Bind::Tex(input),
        Bind::Tex(output),
        Bind::Tex(&blurred),
        Bind::Buf(&params),
      ],
    );
  }
}

pub struct Fonda {
  pub edge_map: EdgeMap,
  pub pop_grade: PopGrade,
  pub abstraction: Abstract,
  pub flatten_colour: FlattenColour,
  pub simplify_texture: SimplifyTexture,
  pub protect_edges: ProtectEdges,
  pub quantize_values: QuantizeValues,
  pub colour_shadows: ColourShadows,
  pub accent_lines: AccentLines,
  pub brush_texture: BrushTexture,
  pub flow_antialias: FlowAntialias,
  pub sharpen_edges: SharpenEdges,
  pub print_finish: PrintFinish,
  context: GpuContext,
  store: ComputePipeline,
}

impl Fonda {
  pub fn new(context: &GpuContext) -> Self {
    Self {
      edge_map: EdgeMap::new(context),
      pop_grade: PopGrade::new(context),
      abstraction: Abstract::new(context),
      flatten_colour: FlattenColour::new(context),
      simplify_texture: SimplifyTexture::new(context),
      protect_edges: ProtectEdges::new(context),
      quantize_values: QuantizeValues::new(context),
      colour_shadows: ColourShadows::new(context),
      accent_lines: AccentLines::new(context),
      brush_texture: BrushTexture::new(context),
      flow_antialias: FlowAntialias::new(context),
      sharpen_edges: SharpenEdges::new(context),
      print_finish: PrintFinish::new(context),
      context: context.clone(),
      store: pipeline(context, "fonda store", STORE),
    }
  }

  fn then<K: Kernel>(
    &self,
    encoder: &mut CommandEncoder,
    kernel: &K,
    input: K::Input<'_>,
    size: Extent3d,
  ) -> Scratch {
    let output = self.context.scratch(size, FORMAT);
    kernel.encode(encoder, input, &output);
    output
  }
}

impl Kernel for Fonda {
  type Input<'a> = &'a Texture;

  fn encode(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture) {
    let size = input.size();

    let edges = self.then(encoder, &self.edge_map, input, size);
    let graded = self.then(encoder, &self.pop_grade, input, size);
    let mut img = self.then(encoder, &self.abstraction, &graded, size);
    img = self.then(encoder, &self.flatten_colour, &img, size);
    img = self.then(encoder, &self.simplify_texture, (&img, input), size);
    let guide = self.then(encoder, &self.protect_edges, (&img, &graded, &edges), size);
    drop((edges, graded));

    img = self.then(encoder, &self.quantize_values, &guide, size);
    img = self.then(encoder, &self.colour_shadows, &img, size);
    img = self.then(encoder, &self.accent_lines, (&img, &guide), size);
    drop(guide);

    img = self.then(encoder, &self.brush_texture, &img, size);
    img = self.then(encoder, &self.flow_antialias, &img, size);
    img = self.then(encoder, &self.sharpen_edges, &img, size);
    img = self.then(encoder, &self.print_finish, &img, size);

    run(
      &self.context,
      encoder,
      &self.store,
      dims(input),
      &[Bind::Tex(&img), Bind::Tex(output), Bind::Tex(input)],
    );
  }
}
