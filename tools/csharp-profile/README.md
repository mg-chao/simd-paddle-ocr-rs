# C# production stage diagnosis

This diagnostic companion to [`cpu_profile`](../../examples/cpu_profile.rs) consumes serial job JSON emitted by `cpu_benchmark`. Its default stage mode uses the job's pinned assets, decoded pixels, settings and affinity, performs **one uninstrumented warm-up and one instrumented sample**, verifies unchanged output, and emits JSON stage wall/work counters. Nested counters overlap. Stage mode does not enable per-node profiling. Every mode preserves production fusion; none executes a diagnostic graph. These timings are not an official performance comparison.

Build with the pinned SDK from this directory, isolating all referenced outputs:

```powershell
$profileArtifacts = [IO.Path]::GetFullPath("../../target/csharp-profile/$([Guid]::NewGuid())")
dotnet build Sdcb.SimdPaddleOCR.Benchmarks.csproj -c Release --nologo --artifacts-path $profileArtifacts -p:ArtifactsPivots=cpu
$env:DOTNET_PROCESSOR_COUNT = '1'
$profileDll = "$profileArtifacts/bin/Sdcb.SimdPaddleOCR.Benchmarks/cpu/Sdcb.SimdPaddleOCR.Benchmarks.dll"
```

Use existing absolute `*-t1-c1.job.json` paths from a successful benchmark input-preparation run. Default stage and node modes require a single-CPU affinity mask, one concurrent request, and `DOTNET_PROCESSOR_COUNT=1`. The adapter verifies fixture dimensions/pixel SHA256 and all four selected model/dictionary hashes before loading. The assembly name uses an existing friend-assembly grant in the pinned C# project; no reference source changes are required.

| Invocation after `$profileDll` | Work performed | Fixed sampling |
|---|---|---|
| `SERIAL_JOB.json` | Full OCR, production stage counters only | One warm-up, one instrumented sample |
| `--kernels` | Three NCHW pointwise shapes and one NHWC depthwise shape | One warm-up invocation and three samples per shape |
| `--recognition-kernels` | Two NHWC recognition pointwise shapes | One warm-up invocation and three single-call samples per shape |
| `--nodes SERIAL_JOB.json` | DET only, production operator/node counters; PipelineProfiler disabled | One warm-up, one instrumented sample |

`--kernels` targets the current AVX2/FMA host. The NCHW dense pointwise cases use 96 input channels, 24 output channels and spatial sizes 19×30, 76×120, 152×240. Each measured sample contains four calls divided by four. The NHWC depthwise case uses batch 1, 96 channels, 152×240 pixels, a 7×7 kernel, stride 1 and padding 3; each sample contains one call. All allocation and weight packing occurs before timers, output storage is reused, and sparse independent scalar checks follow measurements. It chooses the lowest allowed CPU bit and emits every raw sample, milliseconds per call, median, affinity and runtime version.

The pointwise data match Rust's focused probe: `input[i] = ((i*37 % 43)-21)/17`, OIHW weights `((i*19 % 29)-14)/13`, and bias `i/19-0.3`. Depthwise data use `((i*37 + seed*17) % 101)/101-0.5` with seeds 53/59/61 for input/weights/bias. Depthwise input is NHWC; weights start in OIHW order and pass through `Nhwc.PackDepthwise`. The measured entry is `Nhwc.Depthwise`; the NCHW counterpart is `DepthwiseStride1`.

`--nodes` emits the top 30 node names, normalized indices, actual tensor shapes, call counts and milliseconds. C#'s node counters are static and keyed by graph index; restricting execution to DET avoids merging unrelated DET/CLS/REC indices. The report includes the 512-slot counter capacity and model node count. Align nodes by name and shapes when comparing implementations, since inserted layout operations can change indices. Profiling overhead means node/stage totals can differ; neither supplies official latency evidence.

`--recognition-kernels` measures NHWC pointwise products `(rows, inner, columns)`
of `(96,384,768)` and `(960,512,1024)`, matching Rust's ignored
`recognition_pointwise` test. Input and bias use the pointwise formulas above;
the weight formula indexes a row-major `[inner,columns]` matrix, transposed to
OIHW before C# packing. Each sample contains one call, with allocation/packing
outside timing and exact ordered-FMA spot checks afterward. It shares the
ten-second kernel budget and single-CPU/AVX2 requirements of `--kernels`.

Schedule every diagnostic apart from other builds and correctness/performance inference. For a two-model investigation, enforce **25 seconds total for the two stage invocations, 10 seconds for the complete kernel mode, and 25 seconds total for the two node invocations**, excluding build time. The aggregate execution budget is 60 seconds. The kernel mode checks its internal ten-second budget, but every invocation also needs an external process deadline so startup, blocked calls and report generation are bounded. Do not retry failed cases or grow sample counts automatically. Rebuild after source changes and keep different versions' evidence separate.

For example, launch each invocation with a remaining group budget and preserve stdout/stderr:

```powershell
# $profileArguments is @($serialJob), @('--kernels'), or @('--nodes', $serialJob).
# Compute this from the group's remaining 25/10-second budget, also capped by
# the remaining 60-second aggregate budget before every process launch.
$remainingMilliseconds = 10000
$profileCommandArguments = (@($profileDll) + $profileArguments) | ForEach-Object { '"' + $_ + '"' }
$profileProcess = Start-Process dotnet -ArgumentList $profileCommandArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $outputJson -RedirectStandardError $errorLog
if (-not $profileProcess.WaitForExit($remainingMilliseconds)) {
    $profileProcess.Kill($true)
    throw 'Diagnostic deadline exceeded; worker process tree terminated'
}
$profileProcess.WaitForExit()
if ($profileProcess.ExitCode -ne 0) { throw "Diagnostic failed; see $errorLog" }
```

Clear inherited runtime/SIMD overrides before comparison. Stage/node modes reject `PPOCR_*` and `SIMD_OCR_*` overrides; kernel mode checks its required AVX2/FMA capabilities. No mode should inherit CLR SIMD/JIT/profiler overrides that differ from the corresponding Rust/C# benchmark setup.
