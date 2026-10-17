using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;

// Diagnostic companion to cpu_profile: the exact benchmark job/settings/pixels,
// one uninstrumented warm-up and one production-stage instrumented run. This
// friend assembly reads existing counters without modifying the pinned library.
if (args.Length == 1 && args[0] is "--kernels" or "--recognition-kernels")
{
    KernelProbe.Run(args[0] == "--recognition-kernels");
    return;
}
bool nodes = args.Length == 2 && args[0] == "--nodes";
if ((!nodes && args.Length != 1) || !OperatingSystem.IsWindows() || !Environment.Is64BitProcess)
    throw new ArgumentException("usage: Sdcb.SimdPaddleOCR.Benchmarks SERIAL_BENCHMARK_JOB.json (Windows x64)");
foreach (System.Collections.DictionaryEntry entry in Environment.GetEnvironmentVariables())
{
    string name = (string)entry.Key;
    if (name.StartsWith("PPOCR_", StringComparison.OrdinalIgnoreCase) || name.StartsWith("SIMD_OCR_", StringComparison.OrdinalIgnoreCase))
        throw new InvalidDataException($"unset {name} for production-layout/fusion diagnostics");
}
using JsonDocument input = JsonDocument.Parse(File.ReadAllBytes(args[nodes ? 1 : 0]));
JsonElement job = input.RootElement;
if (job.GetProperty("threads").GetInt32() != 1 || job.GetProperty("concurrency").GetInt32() != 1 || Environment.ProcessorCount != 1)
    throw new InvalidDataException("serial diagnostic requires threads=1, concurrency=1 and DOTNET_PROCESSOR_COUNT=1");
string affinity = job.GetProperty("affinity").GetString()!;
ulong mask = Convert.ToUInt64(affinity, 16);
if (System.Numerics.BitOperations.PopCount(mask) != 1) throw new InvalidDataException("expected a single logical CPU affinity mask");
Process.GetCurrentProcess().ProcessorAffinity = unchecked((IntPtr)(long)mask);
int width = job.GetProperty("width").GetInt32(), height = job.GetProperty("height").GetInt32();
byte[] pixels = File.ReadAllBytes(job.GetProperty("pixels").GetString()!);
string pixelsHash = Convert.ToHexStringLower(SHA256.HashData(pixels));
if (pixels.Length != checked(width * height * 3) || pixelsHash != job.GetProperty("pixel_sha256").GetString())
    throw new InvalidDataException("predecoded BGR24 integrity failure");
string manifestPath = job.GetProperty("manifest").GetString()!;
using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
JsonElement settings = manifest.RootElement.GetProperty("settings");
string modelId = job.GetProperty("model").GetString()!, fixtureId = job.GetProperty("fixture").GetString()!;
JsonElement model = manifest.RootElement.GetProperty("models").EnumerateArray().Single(m => m.GetProperty("id").GetString() == modelId);
JsonElement fixture = manifest.RootElement.GetProperty("fixtures").EnumerateArray().Single(f => f.GetProperty("id").GetString() == fixtureId);
if (fixture.GetProperty("pixel_sha256").GetString() != pixelsHash || fixture.GetProperty("width").GetInt32() != width || fixture.GetProperty("height").GetInt32() != height)
    throw new InvalidDataException("pixels do not match the pinned fixture");
string modelsDirectory = job.GetProperty("models").GetString()!;
string ModelPath(string field) => Path.Combine(modelsDirectory, model.GetProperty(field).GetString()!);
foreach (string field in new[] { "det", "cls", "rec", "dictionary" })
{
    string name = model.GetProperty(field).GetString()!;
    JsonElement asset = manifest.RootElement.GetProperty("assets").EnumerateArray().Single(a => a.GetProperty("file").GetString() == name);
    string actual = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(modelsDirectory, name))));
    if (actual != asset.GetProperty("sha256").GetString() && (!asset.TryGetProperty("sha256_crlf", out JsonElement crlf) || actual != crlf.GetString()))
        throw new InvalidDataException($"model/dictionary SHA256 mismatch: {name}");
}
if (nodes)
{
    NodeProbe.Run(ModelPath("det"), pixels, width, height, modelId, fixtureId, affinity, new PaddleOcrDetectorOptions
    {
        Backend = OcrBackend.Cpu,
        LimitSideLength = settings.GetProperty("detector_limit").GetInt32(),
        MaxCandidates = settings.GetProperty("max_candidates").GetInt32(),
        BitmapThreshold = settings.GetProperty("bitmap_threshold").GetSingle(),
        BoxThreshold = model.GetProperty("box_threshold").GetSingle(),
        UnclipRatio = settings.GetProperty("unclip_ratio").GetSingle(),
        UseDilation = settings.GetProperty("use_dilation").GetBoolean(),
    });
    return;
}
Stopwatch load = Stopwatch.StartNew();
using PaddleOcrAll ocr = PaddleOcrAll.Load(ModelPath("det"), ModelPath("cls"), ModelPath("rec"), ModelPath("dictionary"), new PaddleOcrOptions
{
    UseDirectionClassification = settings.GetProperty("direction_classification").GetBoolean(),
    ClassifierThreshold = settings.GetProperty("classifier_threshold").GetSingle(),
    LineWorkerCount = 1, PreprocessWorkerCount = 1, DetIntraOpThreads = 1, RecBatchLines = 1,
    Detector = new PaddleOcrDetectorOptions
    {
        Backend = OcrBackend.Cpu,
        LimitSideLength = settings.GetProperty("detector_limit").GetInt32(),
        MaxCandidates = settings.GetProperty("max_candidates").GetInt32(),
        BitmapThreshold = settings.GetProperty("bitmap_threshold").GetSingle(),
        BoxThreshold = model.GetProperty("box_threshold").GetSingle(),
        UnclipRatio = settings.GetProperty("unclip_ratio").GetSingle(),
        UseDilation = settings.GetProperty("use_dilation").GetBoolean(),
    },
    Classifier = new PaddleOcrClassifierOptions { Backend = OcrBackend.Cpu },
    Recognizer = new PaddleOcrRecognizerOptions
    {
        Backend = OcrBackend.Cpu,
        AdaptiveWidth = settings.GetProperty("adaptive_width").GetBoolean(),
        TargetWidth = settings.GetProperty("target_width").GetInt32(),
    },
});
double loadMs = load.Elapsed.TotalMilliseconds;
Stopwatch watch = Stopwatch.StartNew();
PaddleOcrResult control = ocr.Run(pixels, width, height, width * 3, ImagePixelFormat.Bgr24);
string controlText = control.Text;
double warmupMs = watch.Elapsed.TotalMilliseconds;
PipelineProfiler.Enable(true); // Stage counters only: no per-node instrumentation.
watch.Restart();
PaddleOcrResult result = ocr.Run(pixels, width, height, width * 3, ImagePixelFormat.Bgr24);
string text = result.Text;
double profiledMs = watch.Elapsed.TotalMilliseconds;
var counters = PipelineProfiler.Snapshot();
PipelineProfiler.Enable(false);
if (text != controlText || result.DetectedCount != control.DetectedCount || result.DetectorResizedWidth != control.DetectorResizedWidth
    || result.DetectorResizedHeight != control.DetectorResizedHeight || result.Lines.Length != control.Lines.Length)
    throw new InvalidDataException("profiled output differs from warm-up");
for (int index = 0; index < result.Lines.Length; index++)
{
    PaddleOcrLine a = result.Lines[index], b = control.Lines[index];
    if (a.Box != b.Box || a.Text != b.Text || a.RecognitionScore != b.RecognitionScore || a.ClassificationScore != b.ClassificationScore
        || a.ClassificationLabel != b.ClassificationLabel || a.AppliedRotationDegrees != b.AppliedRotationDegrees || a.EmittedCount != b.EmittedCount
        || a.RecognitionContentWidth != b.RecognitionContentWidth || a.RecognitionTensorWidth != b.RecognitionTensorWidth || a.RecognitionTimeSteps != b.RecognitionTimeSteps)
        throw new InvalidDataException($"profiled line {index} differs from warm-up");
}
Dictionary<string, double> milliseconds = [], calls = [];
for (int index = 0; index < PipelineProfiler.StageCount; index++)
{
    milliseconds[PipelineProfiler.StageNames[index]] = counters[index].Milliseconds;
    calls[PipelineProfiler.StageNames[index]] = counters[index].Calls;
}
Console.WriteLine(JsonSerializer.Serialize(new
{
    instrumented = true, official_benchmark = false, production_fusion = true,
    warmup_runs = 1, measured_runs = 1, model = modelId, fixture = fixtureId, threads = 1, affinity,
    runtime = Environment.Version.ToString(), pixel_sha256 = pixelsHash,
    manifest_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(manifestPath))),
    load_ms = loadMs, warmup_ms = warmupMs, profiled_total_ms = profiledMs,
    detected_lines = result.Lines.Length, output_matches_warmup = true,
    stage_ms = milliseconds, stage_calls = calls,
    note = "Production stage timers only; nested counters overlap. No per-node profiling or diagnostic graph execution; not an official latency comparison.",
}, new JsonSerializerOptions { WriteIndented = true }));
