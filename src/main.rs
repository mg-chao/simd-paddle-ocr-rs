use simd_paddle_ocr_rs::{
    Tolerance,
    cls::run_cls,
    compare_bundles, compare_cls, compare_det, compare_stage, compare_step12,
    det::run_det,
    full_ocr::{compare_ocr, run_ocr},
    model::{self, LoadOptions, Model},
    rec::{compare_rec, run_rec},
    step12::run_step12,
    validate_bundle,
};
use std::{collections::HashSet, env, path::PathBuf, process::ExitCode};

const HELP: &str = "Frozen references, structural ONNX import, and scalar detector/classifier image inference.

Usage:
  simd-paddle-ocr-rs compare <reference> <candidate> [--atol N] [--rtol N] [--json]
  simd-paddle-ocr-rs compare-stage <reference> <candidate> --case ID --artifact ID [--atol N] [--rtol N] [--json]
  simd-paddle-ocr-rs run-cls <reference> --out NEW_DIRECTORY [--json]
  simd-paddle-ocr-rs compare-cls <reference> <candidate> [--atol N] [--rtol N] [--json]
  simd-paddle-ocr-rs run-det <reference> --out NEW_DIRECTORY [--json]
  simd-paddle-ocr-rs compare-det <reference> <candidate> [--atol N] [--rtol N] [--json]
  simd-paddle-ocr-rs run-step12 <reference> --out NEW_DIRECTORY [--json]
  simd-paddle-ocr-rs compare-step12 <reference> <candidate> [--atol N] [--rtol N] [--json]
  simd-paddle-ocr-rs run-rec <reference> --out NEW_DIRECTORY [--json]
  simd-paddle-ocr-rs compare-rec <reference> <candidate> [--atol N] [--rtol N] [--json]
  simd-paddle-ocr-rs run-ocr <reference> --out NEW_DIRECTORY [--json]
  simd-paddle-ocr-rs compare-ocr <reference> <candidate> [--atol N] [--rtol N] [--json]
  simd-paddle-ocr-rs validate <bundle> [--json]
  simd-paddle-ocr-rs inspect-model <model.onnx> [--out NEW_DIRECTORY] [--json]
  simd-paddle-ocr-rs validate-import <reference-directory> [--json]
  simd-paddle-ocr-rs compare-import <reference-directory> <model-id> [--json]
  simd-paddle-ocr-rs --help

Float defaults: --atol 1e-5 --rtol 1e-4.
Shape/layout/order, bytes, text and integer structure compare exactly.
Stage matches cover one selected artifact; every declared file is validated.
CLS matches cover all frozen classifier inputs/nodes/probabilities; no full OCR parity.
DET matches cover frozen normalized detector tensors; no pixel preprocessing.
Step12 matches cover independent pixel-to-oriented-crop inference; no recognition/text.
REC matches cover frozen tiny recognition tensors and both CTC score contracts; no full OCR parity.
OCR matches require complete independent tiny pixel-to-public-result inference.
Import comparison checks raw/normalized graphs and exact weight hashes.
Exit status: 0 match/valid; 1 parity difference; 2 invalid arguments or data.";

struct Arguments {
    command: String,
    paths: Vec<String>,
    case_id: Option<String>,
    artifact_id: Option<String>,
    out: Option<String>,
    tolerance: Tolerance,
    json: bool,
}
fn arguments(args: &[String]) -> Result<Arguments, String> {
    let Some(command) = args.first() else {
        return Err("expected a command".into());
    };
    let mut parsed = Arguments {
        command: command.clone(),
        paths: Vec::new(),
        case_id: None,
        artifact_id: None,
        out: None,
        tolerance: Tolerance::default(),
        json: false,
    };
    let mut seen = HashSet::new();
    let mut index = 1;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag.starts_with('-') && !seen.insert(flag) {
            return Err(format!("duplicate option {flag}"));
        }
        match flag {
            "--json" => parsed.json = true,
            "--atol" | "--rtol" | "--case" | "--artifact" | "--out" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("{flag} needs a value"))?;
                match flag {
                    "--case" => parsed.case_id = Some(value.clone()),
                    "--artifact" => parsed.artifact_id = Some(value.clone()),
                    "--out" => parsed.out = Some(value.clone()),
                    _ => {
                        let number: f64 = value
                            .parse()
                            .map_err(|_| format!("{flag} needs a number"))?;
                        if !number.is_finite() || number < 0.0 {
                            return Err("tolerances must be finite and nonnegative".into());
                        }
                        if flag == "--atol" {
                            parsed.tolerance.atol = number;
                        } else {
                            parsed.tolerance.rtol = number;
                        }
                    }
                }
            }
            unknown if unknown.starts_with('-') => return Err(format!("unknown option {unknown}")),
            path => parsed.paths.push(path.into()),
        }
        index += 1;
    }
    let expected = match parsed.command.as_str() {
        "compare" | "compare-stage" | "compare-cls" | "compare-det" | "compare-step12"
        | "compare-rec" | "compare-ocr" | "compare-import" => 2,
        "validate" | "inspect-model" | "validate-import" | "run-cls" | "run-det" | "run-step12"
        | "run-rec" | "run-ocr" => 1,
        other => return Err(format!("unknown command {other}")),
    };
    if parsed.paths.len() != expected {
        return Err(format!(
            "{} expects {expected} positional arguments",
            parsed.command
        ));
    }
    if parsed.command == "compare-stage" {
        if parsed.case_id.as_deref().is_none_or(str::is_empty)
            || parsed.artifact_id.as_deref().is_none_or(str::is_empty)
        {
            return Err("compare-stage requires nonempty --case and --artifact".into());
        }
    } else if parsed.case_id.is_some() || parsed.artifact_id.is_some() {
        return Err("--case/--artifact require compare-stage".into());
    }
    if parsed.out.is_some()
        && !matches!(
            parsed.command.as_str(),
            "inspect-model" | "run-cls" | "run-det" | "run-step12" | "run-rec" | "run-ocr"
        )
    {
        return Err("--out requires inspect-model or a run command".into());
    }
    if matches!(
        parsed.command.as_str(),
        "run-cls" | "run-det" | "run-step12" | "run-rec" | "run-ocr"
    ) && parsed.out.as_deref().is_none_or(str::is_empty)
    {
        return Err(format!("{} requires nonempty --out", parsed.command));
    }
    if parsed.command != "compare"
        && parsed.command != "compare-stage"
        && parsed.command != "compare-cls"
        && parsed.command != "compare-det"
        && parsed.command != "compare-step12"
        && parsed.command != "compare-rec"
        && parsed.command != "compare-ocr"
        && (seen.contains("--atol") || seen.contains("--rtol"))
    {
        return Err("--atol/--rtol require a tensor comparison command".into());
    }
    Ok(parsed)
}
fn print_report<T: serde::Serialize + std::fmt::Display>(
    report: &T,
    json: bool,
) -> Result<(), String> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(report).map_err(|e| e.to_string())?
        );
    } else {
        println!("{report}");
    }
    Ok(())
}
fn run(args: Arguments) -> Result<u8, String> {
    let path = PathBuf::from(&args.paths[0]);
    match args.command.as_str() {
        "compare" | "compare-stage" | "compare-cls" | "compare-det" | "compare-step12"
        | "compare-rec" | "compare-ocr" => {
            let other = PathBuf::from(&args.paths[1]);
            let report = if args.command == "compare-stage" {
                compare_stage(
                    &path,
                    &other,
                    args.case_id.as_deref().unwrap(),
                    args.artifact_id.as_deref().unwrap(),
                    args.tolerance,
                )
            } else if args.command == "compare-cls" {
                compare_cls(&path, &other, args.tolerance)
            } else if args.command == "compare-det" {
                compare_det(&path, &other, args.tolerance)
            } else if args.command == "compare-step12" {
                compare_step12(&path, &other, args.tolerance)
            } else if args.command == "compare-rec" {
                compare_rec(&path, &other, args.tolerance)
            } else if args.command == "compare-ocr" {
                compare_ocr(&path, &other, args.tolerance)
            } else {
                compare_bundles(&path, &other, args.tolerance)
            }
            .map_err(|e| e.to_string())?;
            print_report(&report, args.json)?;
            Ok(u8::from(!report.matches()))
        }
        "run-cls" => {
            print_report(
                &run_cls(&path, &PathBuf::from(args.out.as_ref().unwrap()))
                    .map_err(|e| e.to_string())?,
                args.json,
            )?;
            Ok(0)
        }
        "run-det" => {
            print_report(
                &run_det(&path, &PathBuf::from(args.out.as_ref().unwrap()))
                    .map_err(|e| e.to_string())?,
                args.json,
            )?;
            Ok(0)
        }
        "run-step12" => {
            print_report(
                &run_step12(&path, &PathBuf::from(args.out.as_ref().unwrap()))
                    .map_err(|e| e.to_string())?,
                args.json,
            )?;
            Ok(0)
        }
        "run-rec" => {
            print_report(
                &run_rec(&path, &PathBuf::from(args.out.as_ref().unwrap()))
                    .map_err(|e| e.to_string())?,
                args.json,
            )?;
            Ok(0)
        }
        "run-ocr" => {
            print_report(
                &run_ocr(&path, &PathBuf::from(args.out.as_ref().unwrap()))
                    .map_err(|e| e.to_string())?,
                args.json,
            )?;
            Ok(0)
        }
        "validate" => {
            print_report(
                &validate_bundle(&path).map_err(|e| e.to_string())?,
                args.json,
            )?;
            Ok(0)
        }
        "inspect-model" => {
            let options = LoadOptions {
                capture_raw_snapshot: true,
                ..Default::default()
            };
            if let Some(directory) = &args.out {
                print_report(
                    &model::inspect_model(&path, directory, &options).map_err(|e| e.to_string())?,
                    args.json,
                )?;
            } else {
                let imported =
                    Model::load_path_with_options(&path, &options).map_err(|e| e.to_string())?;
                if args.json {
                    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                        "source_bytes": imported.source_bytes(),
                        "raw": imported.raw_snapshot(),
                        "normalized": imported.normalized_snapshot().map_err(|e| e.to_string())?,
                        "shape_recipes": imported.shape_recipes(),
                    })).map_err(|e| e.to_string())?);
                } else {
                    println!(
                        "IMPORTED: {} nodes, {} tensors; {} source bytes, {} retained weight bytes",
                        imported.graph().nodes.len(),
                        imported.graph().tensors.len(),
                        imported.source_bytes(),
                        imported.normalized_weight_bytes()
                    );
                }
            }
            Ok(0)
        }
        "validate-import" => {
            print_report(
                &model::validate_import_reference(&path).map_err(|e| e.to_string())?,
                args.json,
            )?;
            Ok(0)
        }
        "compare-import" => {
            let report = model::compare_import(&path, &args.paths[1], &LoadOptions::default())
                .map_err(|e| e.to_string())?;
            print_report(&report, args.json)?;
            Ok(u8::from(!report.matches()))
        }
        _ => unreachable!(),
    }
}
fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if matches!(args.as_slice(), [help] if help == "--help" || help == "-h" || help == "help") {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let parsed = match arguments(&args) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("Invalid arguments: {error}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    match run(parsed) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("Invalid data: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(args: &[&str]) -> Result<Arguments, String> {
        arguments(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }
    #[test]
    fn new_commands_reject_ambiguous_arguments() {
        assert!(
            parse(&[
                "compare-stage",
                "a",
                "b",
                "--case",
                "sample",
                "--artifact",
                "det.input"
            ])
            .is_ok()
        );
        assert!(parse(&["compare-stage", "a", "b", "--case", "sample"]).is_err());
        assert!(parse(&["validate", "a", "--rtol", "1"]).is_err());
        assert!(parse(&["compare-import", "a", "cls", "--artifact", "x"]).is_err());
        assert!(parse(&["validate", "a", "--json", "--json"]).is_err());
        assert!(parse(&["compare", "a", "b", "--atol", "0", "--rtol", "0"]).is_ok());
        assert!(parse(&["run-cls", "reference", "--out", "fresh", "--json"]).is_ok());
        assert!(parse(&["run-cls", "reference"]).is_err());
        assert!(parse(&["run-cls", "reference", "--out", "fresh", "--atol", "1"]).is_err());
        assert!(parse(&["compare-cls", "reference", "candidate", "--atol", "0"]).is_ok());
        assert!(parse(&["compare-cls", "reference", "candidate", "--case", "sample"]).is_err());
    }
}
