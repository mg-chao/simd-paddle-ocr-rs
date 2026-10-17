#!/bin/bash
# Local correctness validation; model/reference setup is in docs/macos-cpu-validation.md.
set -euo pipefail

ocr_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ocr_rust_only=false
ocr_paths=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rust-only) ocr_rust_only=true; shift ;;
        --reference-root|--models-dir)
            if [[ $# -lt 2 || -z "$2" ]]; then
                printf 'Missing path for %s\n' "$1" >&2
                exit 1
            fi
            # Resolve existing directories before changing to the repository.
            ocr_paths+=("$1" "$(cd -- "$2" && pwd)")
            shift 2
            ;;
        -h|--help)
            printf '%s\n' \
                'Usage: bash tools/validate-macos.sh [--rust-only] [--reference-root PATH] [--models-dir PATH]' \
                'Runs release builds, default/serial tests, the 66-case baseline, and full 213-case qualification.' \
                '--rust-only omits live C# comparisons and configuration qualification; provide local models.'
            exit 0
            ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

if [[ "$(uname -s)" != Darwin ]]; then
    printf '%s\n' 'Run this validation on a physical macOS machine.' >&2
    exit 1
fi
cd -- "$ocr_root"
sw_vers
uname -m
rustc -vV
if ! $ocr_rust_only; then
    # Run in the adapter directory so its exact SDK pin is honored.
    (cd tools/csharp-reference && dotnet --version)
fi
cargo fmt --all -- --check
cargo build --release --locked --examples
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
if $ocr_rust_only; then
    # Bash 3.2 (shipped with macOS) treats empty arrays as unset under nounset.
    cargo run --release --locked --example cpu_baseline -- verify --rust-only ${ocr_paths[@]+"${ocr_paths[@]}"}
else
    cargo run --release --locked --example cpu_baseline -- verify ${ocr_paths[@]+"${ocr_paths[@]}"}
    cargo run --release --locked --example cpu_qualify -- ${ocr_paths[@]+"${ocr_paths[@]}"}
fi
