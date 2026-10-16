# simd-paddle-ocr-rs

Rust CPU implementation of SimdPaddleOCR, with portable SIMD and runtime CPU
dispatch. The pinned Rust toolchain is selected by `rust-toolchain.toml`.

## Correctness baseline

See [the maintained CPU baseline guide](docs/cpu-baseline.md) for the pinned
C# reference, model/dictionary SHA-256 hashes, 22 public test fixtures, saved
tiny/small/medium predictions, and release verification commands.

```powershell
cargo build --release --locked --examples
cargo test --release --locked
cargo run --release --locked --example cpu_baseline -- verify
```

The live comparison expects the pinned C# checkout and its models in the
sibling `../SimdPaddleOCR` directory. The guide also describes offline
Rust-only verification and intentional baseline regeneration.
