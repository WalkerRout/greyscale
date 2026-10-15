use wgpu::{
  BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, CommandEncoder,
  ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, ShaderModule,
  ShaderModuleDescriptor, ShaderSource, Texture,
};

use crate::gpu::{self, GpuContext};

pub mod box_blur;
pub mod edge_detection;
pub mod fonda;
pub mod gaussian_blur;
pub mod greyscale;

pub const WORKGROUP_SIZE: (u32, u32) = (16, 16);

pub trait Kernel {
  type Input<'a>;

  fn encode(&self, encoder: &mut CommandEncoder, input: Self::Input<'_>, output: &Texture);
}

pub fn create_shader_module(ctx: &GpuContext, label: &str, shader_source: &str) -> ShaderModule {
  ctx.device.create_shader_module(ShaderModuleDescriptor {
    label: Some(label),
    source: ShaderSource::Wgsl(shader_source.into()),
  })
}

pub fn create_compute_pipeline(
  ctx: &GpuContext,
  label: &str,
  module: &ShaderModule,
  entry_point: &str,
) -> ComputePipeline {
  ctx
    .device
    .create_compute_pipeline(&ComputePipelineDescriptor {
      label: Some(label),
      module,
      entry_point: Some(entry_point),
      compilation_options: Default::default(),
      cache: None,
      layout: None,
    })
}

pub fn create_bind_group(
  ctx: &GpuContext,
  label: &str,
  pipeline: &ComputePipeline,
  resources: &[BindingResource],
) -> BindGroup {
  let entries: Vec<BindGroupEntry> = resources
    .iter()
    .cloned()
    .enumerate()
    .map(|(binding, resource)| BindGroupEntry {
      binding: binding as u32,
      resource,
    })
    .collect();

  ctx.device.create_bind_group(&BindGroupDescriptor {
    label: Some(label),
    layout: &pipeline.get_bind_group_layout(0),
    entries: &entries,
  })
}

pub fn dispatch(
  encoder: &mut CommandEncoder,
  pipeline: &ComputePipeline,
  bind_group: &BindGroup,
  (width, height): (u32, u32),
) {
  let (x, y) = gpu::compute_work_group_count((width, height), WORKGROUP_SIZE);
  let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
    label: Some("compute pass"),
    timestamp_writes: Default::default(),
  });
  compute_pass.set_pipeline(pipeline);
  compute_pass.set_bind_group(0, bind_group, &[]);
  compute_pass.dispatch_workgroups(x, y, 1);
}
