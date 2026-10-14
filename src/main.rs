use std::{env, path::PathBuf, process::ExitCode};

use simd_paddle_ocr_rs::{Tolerance, compare_bundles};

const HELP: &str = "Compare frozen OCR reference bundles.

Usage:
  simd-paddle-ocr-rs compare <reference-dir> <candidate-dir> [--atol N] [--rtol N]
  simd-paddle-ocr-rs --help

Defaults: --atol 1e-5 --rtol 1e-4
For float tensors and JSON numbers, |candidate-reference| <= atol + rtol*|reference|.
JSON integer pairs, strings, booleans, keys, and array order compare exactly.
An integer compared with a floating-point JSON number uses the float tolerance.
Case/artifact order, dtype, shape, layout, contracts, and model hashes must match.
Every model/artifact file is checked against its recorded hash and byte length first.
Exit status: 0 match; 1 parity difference; 2 invalid arguments or bundle.";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.as_slice() == ["--help"] || args.as_slice() == ["-h"] || args.as_slice() == ["help"] {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let (reference, candidate, tolerance) = match parse_args(&args) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("Invalid arguments: {error}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    match compare_bundles(&reference, &candidate, tolerance) {
        Ok(report) => {
            println!("{report}");
            ExitCode::from(u8::from(!report.matches()))
        }
        Err(error) => {
            eprintln!("Invalid bundle: {error}");
            ExitCode::from(2)
        }
    }
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, Tolerance), String> {
    if args.first().map(String::as_str) != Some("compare") {
        return Err("expected the 'compare' command".into());
    }
    let mut directories = Vec::new();
    let mut tolerance = Tolerance::default();
    let mut index = 1;
    let mut seen_atol = false;
    let mut seen_rtol = false;
    while index < args.len() {
        match args[index].as_str() {
            flag @ ("--atol" | "--rtol") => {
                let seen = if flag == "--atol" {
                    &mut seen_atol
                } else {
                    &mut seen_rtol
                };
                if *seen {
                    return Err(format!("duplicate option {flag}"));
                }
                *seen = true;
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("{flag} needs a number"))?
                    .parse::<f64>()
                    .map_err(|_| format!("{flag} needs a number"))?;
                if !value.is_finite() || value < 0.0 {
                    return Err(format!("{flag} must be finite and nonnegative"));
                }
                if flag == "--atol" {
                    tolerance.atol = value;
                } else {
                    tolerance.rtol = value;
                }
            }
            flag if flag.starts_with('-') => return Err(format!("unknown option {flag}")),
            directory => directories.push(PathBuf::from(directory)),
        }
        index += 1;
    }
    if directories.len() != 2 {
        return Err("expected exactly two bundle directories".into());
    }
    Ok((directories.remove(0), directories.remove(0), tolerance))
}
