using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Sdcb.SimdPaddleOCR;

// No profiling switches, timers inside production code, image decode, or JSON work
// in the measured region. The parent pins inputs, provenance, build, and deadlines.
if (args.Length != 1 || !OperatingSystem.IsWindows() || !Environment.Is64BitProcess)
    throw new ArgumentException("Benchmark requires Windows x64 and one job JSON path");
JsonSerializerOptions json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow };
Job job = JsonSerializer.Deserialize<Job>(File.ReadAllBytes(args[0]), json)!;
if (job.Threads is not (1 or 8) || job.Concurrency is not (1 or 2) || job.Runs is < 1 or > 3 || Environment.ProcessorCount != job.Threads)
    throw new InvalidDataException("invalid worker budget or DOTNET_PROCESSOR_COUNT");
using Process benchmarkProcess = Process.GetCurrentProcess();
benchmarkProcess.ProcessorAffinity = unchecked((IntPtr)(long)Convert.ToUInt64(job.Affinity, 16));
byte[] pixels = File.ReadAllBytes(job.Pixels);
if (pixels.Length != checked(job.Width * job.Height * 3) || Convert.ToHexStringLower(SHA256.HashData(pixels)) != job.PixelSha256)
    throw new InvalidDataException("predecoded BGR24 integrity failure");
using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(job.Manifest));
JsonElement settings = manifest.RootElement.GetProperty("settings");
JsonElement model = manifest.RootElement.GetProperty("models").EnumerateArray().Single(m => m.GetProperty("id").GetString() == job.Model);
string ModelPath(string field) => Path.Combine(job.Models, model.GetProperty(field).GetString()!);
Stopwatch load = Stopwatch.StartNew();
using PaddleOcrModelSet models = PaddleOcrModelSet.Load(ModelPath("det"), ModelPath("cls"), ModelPath("rec"), ModelPath("dictionary"));
using PaddleOcrAll ocr = new(models, new PaddleOcrOptions
{
    UseDirectionClassification = settings.GetProperty("direction_classification").GetBoolean(),
    ClassifierThreshold = settings.GetProperty("classifier_threshold").GetSingle(),
    LineWorkerCount = job.Threads == 1 ? 1 : 4,
    PreprocessWorkerCount = job.Threads,
    DetIntraOpThreads = job.Threads,
    RecBatchLines = 1,
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
double loadCompileMs = load.Elapsed.TotalMilliseconds;
Console.Error.WriteLine($".NET={Environment.Version}; CPU={Environment.ProcessorCount}; line={ocr.EffectiveLineWorkerCount}; affinity={job.Affinity}; AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported}; FMA={System.Runtime.Intrinsics.X86.Fma.IsSupported}; AVX512={System.Runtime.Intrinsics.X86.Avx512F.IsSupported}; owned CPU results; no profiling");
(double Milliseconds, double ProcessCpuMilliseconds, OwnedResult[] Results) Batch()
{
    PaddleOcrResult[] results = new PaddleOcrResult[job.Concurrency];
    Exception? failure = null;
    TimeSpan cpuStart = benchmarkProcess.TotalProcessorTime;
    Stopwatch watch = Stopwatch.StartNew();
    if (job.Concurrency == 1) results[0] = ocr.Run(pixels, job.Width, job.Height, job.Width * 3, ImagePixelFormat.Bgr24);
    else
    {
        using Barrier barrier = new(job.Concurrency);
        Thread[] workers = Enumerable.Range(0, job.Concurrency).Select(index => new Thread(() =>
        {
            try { barrier.SignalAndWait(); results[index] = ocr.Run(pixels, job.Width, job.Height, job.Width * 3, ImagePixelFormat.Bgr24); }
            catch (Exception e) { Interlocked.CompareExchange(ref failure, e, null); }
        })).ToArray();
        foreach (Thread thread in workers) thread.Start();
        foreach (Thread thread in workers) thread.Join();
    }
    if (failure is not null) throw new InvalidOperationException("concurrent request failed", failure);
    // Text is lazy in C#; materialize and retain it inside the timer to match
    // Rust's owned aggregate output, including the joined text string.
    OwnedResult[] owned = results.Select(result => new OwnedResult(result, result.Text)).ToArray();
    double elapsed = watch.Elapsed.TotalMilliseconds;
    double cpuElapsed = (benchmarkProcess.TotalProcessorTime - cpuStart).TotalMilliseconds;
    if (!double.IsFinite(cpuElapsed) || cpuElapsed < 0) throw new InvalidDataException("invalid process CPU time");
    GC.KeepAlive(owned);
    return (elapsed, cpuElapsed, owned);
}
bool Same(OwnedResult a, OwnedResult b)
{
    if (a.Text != b.Text || a.Value.DetectedCount != b.Value.DetectedCount || a.Value.DetectorResizedWidth != b.Value.DetectorResizedWidth
        || a.Value.DetectorResizedHeight != b.Value.DetectorResizedHeight || a.Value.Lines.Length != b.Value.Lines.Length) return false;
    for (int index = 0; index < a.Value.Lines.Length; index++)
    {
        PaddleOcrLine x = a.Value.Lines[index], y = b.Value.Lines[index];
        if (x.Box != y.Box || x.Text != y.Text || x.RecognitionScore != y.RecognitionScore || x.ClassificationScore != y.ClassificationScore
            || x.ClassificationLabel != y.ClassificationLabel || x.AppliedRotationDegrees != y.AppliedRotationDegrees || x.EmittedCount != y.EmittedCount
            || x.RecognitionContentWidth != y.RecognitionContentWidth || x.RecognitionTensorWidth != y.RecognitionTensorWidth || x.RecognitionTimeSteps != y.RecognitionTimeSteps)
            return false;
    }
    return true;
}
var first = Batch();
double firstInferenceMs = first.Milliseconds;
double firstCpuMs = first.ProcessCpuMilliseconds;
OwnedResult control = first.Results[0];
if (first.Results.Any(result => !Same(control, result))) throw new InvalidDataException("concurrent cold C# results differ");
first = default;
var warm = Batch();
double warmupMs = warm.Milliseconds;
double warmupCpuMs = warm.ProcessCpuMilliseconds;
if (warm.Results.Any(result => !Same(control, result))) throw new InvalidDataException("C# warm-up result differs from cold control");
warm = default;
List<double> samplesMs = [];
List<double> samplesCpuMs = [];
OwnedResult[] finalResults = [];
for (int index = 0; index < job.Runs; index++)
{
    finalResults = [];
    var measured = Batch();
    if (measured.Results.Any(result => !Same(control, result))) throw new InvalidDataException("C# measured result differs from cold control");
    samplesMs.Add(measured.Milliseconds);
    samplesCpuMs.Add(measured.ProcessCpuMilliseconds);
    finalResults = measured.Results;
}
// Capture official peak before reflection, projection, JSON or diagnostic allocation.
ulong peakPrivateBytes = WindowsMemory.PeakCommit();
object Project(OwnedResult owned)
{
    PaddleOcrResult result = owned.Value;
    return new {
    Id = job.Fixture, owned.Text, result.DetectedCount, result.DetectorResizedWidth, result.DetectorResizedHeight,
    Lines = result.Lines.Select(l => new
    {
        Points = new[] { new[] { l.Box.X1, l.Box.Y1 }, new[] { l.Box.X2, l.Box.Y2 }, new[] { l.Box.X3, l.Box.Y3 }, new[] { l.Box.X4, l.Box.Y4 } },
        DetectionScore = l.Box.Score, l.Text, l.RecognitionScore, l.ClassificationScore, l.ClassificationLabel,
        l.AppliedRotationDegrees, l.EmittedCount, l.RecognitionContentWidth, l.RecognitionTensorWidth, l.RecognitionTimeSteps,
    }).ToArray(),
    };
}
Dictionary<string, ulong> diagnostics = RetainedDiagnostics.Inspect(ocr);
diagnostics["source_weight_bytes"] = models.DetectionModel.Info.WeightSize + models.RecognitionModel.Info.WeightSize + (models.ClassificationModel?.Info.WeightSize ?? 0);
diagnostics["concurrent_owned_results"] = (ulong)job.Concurrency;
Console.WriteLine(JsonSerializer.Serialize(new
{
    Language = "csharp", LoadCompileMs = loadCompileMs, FirstInferenceMs = firstInferenceMs, WarmupMs = warmupMs,
    SamplesMs = samplesMs,
    ProcessCpu = new { FirstInferenceMs = firstCpuMs, WarmupMs = warmupCpuMs, SamplesMs = samplesCpuMs },
    PeakPrivateBytes = peakPrivateBytes, Diagnostics = diagnostics, Result = Project(finalResults[0]),
}, json));

record Job(string Manifest, string Models, string Model, string Fixture, string Pixels, string PixelSha256,
    int Width, int Height, int Threads, int Concurrency, int Runs, string Affinity);
record OwnedResult(PaddleOcrResult Value, string Text);

static class WindowsMemory
{
    [StructLayout(LayoutKind.Sequential)]
    struct Counters
    {
        public uint Size, Faults;
        public nuint PeakWorkingSet, WorkingSet, PeakPagedPool, PagedPool, PeakNonpagedPool, NonpagedPool,
            Pagefile, PeakPagefile, Private;
    }
    [DllImport("psapi.dll", SetLastError = true)]
    static extern bool GetProcessMemoryInfo(IntPtr process, ref Counters counters, uint size);
    public static ulong PeakCommit()
    {
        Counters counters = new() { Size = (uint)Marshal.SizeOf<Counters>() };
        if (!GetProcessMemoryInfo(Process.GetCurrentProcess().Handle, ref counters, counters.Size))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        return (ulong)counters.PeakPagefile;
    }
}

// Pinned-reference diagnostics only: count reachable native activation capacities,
// primitive request arrays and cached packs after all measured work. These are
// deliberately labeled lower bounds, exclude headers/TLS/shared ArrayPool buffers,
// and never participate in the official process-memory comparison.
static class RetainedDiagnostics
{
    public static Dictionary<string, ulong> Inspect(object root)
    {
        HashSet<object> seen = new(ReferenceEqualityComparer.Instance);
        Dictionary<string, ulong> counts = new()
        {
            ["retained_activation_bytes_lower_bound"] = 0,
            ["retained_primitive_request_array_bytes_lower_bound"] = 0,
            ["retained_resize_scratch_bytes_lower_bound"] = 0,
            ["retained_inference_sessions"] = 0,
            ["shared_cached_weight_bytes_lower_bound"] = 0,
        };
        void Walk(object? value, bool resize = false, bool packs = false)
        {
            if (value is null || value is string || !seen.Add(value)) return;
            Type type = value.GetType();
            if (type.IsPrimitive || type.IsEnum || type.IsPointer) return;
            string name = type.FullName ?? "";
            if (name == "Sdcb.SimdPaddleOCR.OnnxSharp.NativeWorkspace")
            {
                counts["retained_activation_bytes_lower_bound"] += checked((ulong)(int)type.GetProperty("Length")!.GetValue(value)! * 4);
                return;
            }
            if (name == "Sdcb.SimdPaddleOCR.OnnxSharp.InferenceSession") counts["retained_inference_sessions"]++;
            resize |= name.Contains("ResizeWorkspace");
            if (name == "Sdcb.SimdPaddleOCR.OnnxSharp.Model")
            {
                Walk(type.GetField("_packedWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(value), false, true);
                return;
            }
            if (name == "Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel")
            {
                Walk(type.GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(value));
                return; // Shared constant arrays and planning metadata are not request buffers.
            }
            if (name == "Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel+TensorMeta") return;
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Lazy<>))
            {
                if ((bool)type.GetProperty("IsValueCreated")!.GetValue(value)!) Walk(type.GetProperty("Value")!.GetValue(value), resize, packs);
                return;
            }
            if (value is Array array)
            {
                if (type.GetElementType()!.IsPrimitive)
                {
                    ulong bytes = checked((ulong)Buffer.ByteLength(array));
                    if (packs) counts["shared_cached_weight_bytes_lower_bound"] += bytes;
                    else { counts["retained_primitive_request_array_bytes_lower_bound"] += bytes; if (resize) counts["retained_resize_scratch_bytes_lower_bound"] += bytes; }
                }
                else foreach (object? item in array) Walk(item, resize, packs);
                return;
            }
            if (value is IEnumerable sequence)
            {
                foreach (object? item in sequence)
                {
                    if (item is not null && item.GetType().IsGenericType && item.GetType().GetGenericTypeDefinition() == typeof(KeyValuePair<,>))
                        Walk(item.GetType().GetProperty("Value")!.GetValue(item), resize, packs);
                    else Walk(item, resize, packs);
                }
                return;
            }
            if (!name.StartsWith("Sdcb.SimdPaddleOCR")) return;
            foreach (FieldInfo field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
                if (!field.FieldType.IsPointer && !field.FieldType.IsPrimitive && !field.FieldType.IsEnum) Walk(field.GetValue(value), resize, packs);
        }
        Walk(root);
        return counts;
    }
}
