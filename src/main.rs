use pollster::FutureExt;

mod gpu;
mod process;
mod shader;

use gpu::GpuContext;

use process::ImageProcessor;

use shader::box_blur::BoxBlur;
use shader::edge_detection::EdgeDetection;
use shader::gaussian_blur::GaussianBlur;
use shader::greyscale::Greyscale;

fn main() -> Result<(), anyhow::Error> {
  let gpu_context = GpuContext::new().block_on()?;

  let image = image::load_from_memory(include_bytes!("../fractal.png"))?.into_rgba8();
  let (width, height) = image.dimensions();
  println!("Processing image: {}x{}", width, height);

  let processor = ImageProcessor::new(&gpu_context);

  {
    let grayscale = Greyscale::new(&gpu_context);
    let output = processor.process_image(&image, &grayscale)?;
    output.save("grayscale.png")?;
    println!("Saved grayscale.png");
  }

  {
    let gaussian_blur = GaussianBlur::new(&gpu_context, 9);
    let output = processor.process_image(&image, &gaussian_blur)?;
    output.save("gaussian_blur.png")?;
    println!("Saved gaussian_blur.png");
  }

  {
    let box_blur = BoxBlur::new(&gpu_context, 5);
    let output = processor.process_image(&image, &box_blur)?;
    output.save("box_blur.png")?;
    println!("Saved box_blur.png");
  }

  {
    let edge_detection = EdgeDetection::new(&gpu_context);
    let output = processor.process_image(&image, &edge_detection)?;
    output.save("edge_detection.png")?;
    println!("Saved edge_detection.png");
  }

  Ok(())
}
