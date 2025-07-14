use wgpu::{
  BindGroup, ComputePipeline, ComputePipelineDescriptor, ShaderModule, ShaderModuleDescriptor,
  ShaderSource, TextureView,
};

use crate::gpu::GpuContext;

pub mod box_blur;
pub mod edge_detection;
pub mod gaussian_blur;
pub mod greyscale;

pub trait Kernel {
  /// Fetch the compute pipeline we want to use for the encoder's compute pass
  fn pipeline(&self) -> &ComputePipeline;

  /// Our kernel must be able to create a bind group operating over one source
  /// and one destination texture...
  fn create_bind_group(&self, input: &TextureView, output: &TextureView) -> BindGroup;

  /// Default 256 threads/wg
  fn workgroup_size(&self) -> (u32, u32) {
    (16, 16)
  }
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
