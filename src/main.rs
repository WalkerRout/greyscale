use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use argh::FromArgs;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Filter {
  Greyscale,
  GaussianBlur,
  BoxBlur,
  EdgeDetection,
}

impl Filter {
  const ALL: [Filter; 4] = [
    Filter::Greyscale,
    Filter::GaussianBlur,
    Filter::BoxBlur,
    Filter::EdgeDetection,
  ];
}

impl fmt::Display for Filter {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Filter::Greyscale => write!(f, "greyscale"),
      Filter::GaussianBlur => write!(f, "gaussian-blur"),
      Filter::BoxBlur => write!(f, "box-blur"),
      Filter::EdgeDetection => write!(f, "edge-detection"),
    }
  }
}

impl argh::FromArgValue for Filter {
  fn from_arg_value(value: &str) -> Result<Self, String> {
    Filter::ALL
      .into_iter()
      .find(|filter| filter.to_string() == value)
      .ok_or_else(|| {
        let valid: Vec<String> = Filter::ALL.iter().map(|f| f.to_string()).collect();
        format!(
          "unknown kernel '{value}', expected one of '{}'",
          valid.join(", ")
        )
      })
  }
}

#[derive(FromArgs)]
#[argh(description = "greyscale - run gpu image kernels over a picture")]
struct Args {
  #[argh(positional, description = "input image file")]
  input: String,

  #[argh(
    option,
    short = 'o',
    description = "output image path, or a directory to write <stem>.<kernel>.png into (default: <stem>.<kernel>.png)"
  )]
  output: Option<String>,

  #[argh(
    option,
    short = 'k',
    default = "Filter::Greyscale",
    description = "kernel: greyscale, gaussian-blur, box-blur, or edge-detection (default: greyscale)"
  )]
  kernel: Filter,

  #[argh(
    option,
    short = 'r',
    description = "blur radius in pixels (default: 9 for gaussian-blur, 5 for box-blur)"
  )]
  radius: Option<u32>,
}

fn main() -> Result<(), anyhow::Error> {
  let args: Args = argh::from_env();

  if args.radius.is_some() && !matches!(args.kernel, Filter::GaussianBlur | Filter::BoxBlur) {
    anyhow::bail!("--radius only applies to gaussian-blur and box-blur");
  }

  let image = image::open(&args.input)?.into_rgba8();
  let (width, height) = image.dimensions();
  eprintln!("processing {}: {width}x{height}", args.input);

  let gpu_context = GpuContext::new().block_on()?;
  let processor = ImageProcessor::new(&gpu_context);

  let output = match args.kernel {
    Filter::Greyscale => processor.process_image(&image, &Greyscale::new(&gpu_context))?,
    Filter::GaussianBlur => {
      let kernel = GaussianBlur::new(&gpu_context, args.radius.unwrap_or(9));
      processor.process_image(&image, &kernel)?
    }
    Filter::BoxBlur => {
      let kernel = BoxBlur::new(&gpu_context, args.radius.unwrap_or(5));
      processor.process_image(&image, &kernel)?
    }
    Filter::EdgeDetection => processor.process_image(&image, &EdgeDetection::new(&gpu_context))?,
  };

  let stem = Path::new(&args.input)
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("output");
  let default_name = format!("{stem}.{}.png", args.kernel);

  let output_path = match args.output {
    Some(out) if out.ends_with(['/', '\\']) || Path::new(&out).is_dir() => {
      fs::create_dir_all(&out)?;
      Path::new(&out).join(default_name)
    }
    Some(out) => PathBuf::from(out),
    None => PathBuf::from(default_name),
  };

  output.save(&output_path)?;
  eprintln!("saved: {}", output_path.display());

  Ok(())
}
