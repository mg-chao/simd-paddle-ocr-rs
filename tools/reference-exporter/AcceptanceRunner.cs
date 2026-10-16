using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

internal static partial class AcceptanceRunner
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
    private static Dictionary<string, string?> JitEnvironment()
    {
        string[] suffixes = ["TIEREDCOMPILATION", "TIEREDPGO", "TC_QUICKJIT", "TC_QUICKJITFORLOOPS", "TC_ONSTACKREPLACEMENT", "READYTORUN"];
        Dictionary<string, string?> values = new(StringComparer.Ordinal);
        foreach (string prefix in new[] { "DOTNET_", "COMPLUS_" })
            foreach (string suffix in suffixes) values.Add(prefix + suffix, null);
        foreach (DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            string name = ((string)entry.Key).ToUpperInvariant();
            if (!name.StartsWith("DOTNET_", StringComparison.Ordinal) && !name.StartsWith("COMPLUS_", StringComparison.Ordinal)) continue;
            string suffix = name[(name.IndexOf('_') + 1)..];
            if (new[] { "TIERED", "PGO", "TC_", "QUICKJIT", "READYTORUN", "JIT" }.Any(prefix => suffix.StartsWith(prefix, StringComparison.Ordinal)))
                values[name] = (string?)entry.Value;
        }
        return values;
    }
    private sealed record Image(string Id, int Width, int Height, byte[] Bytes);
    private sealed record Work(Image Image, int Iterations);
    private sealed record Response(PaddleOcrResult? Result, Exception? Error);
    private sealed class Workers : IDisposable
    {
        private readonly List<BlockingCollection<Work>> _commands = [];
        private readonly BlockingCollection<Response> _responses = new();
        private readonly List<Thread> _threads = [];
        private readonly Barrier _ready, _start, _finish;
        internal Workers(PaddleOcrAll pipeline, int count)
        {
            _ready = new(count + 1); _start = new(count + 1); _finish = new(count + 1);
            for (int i = 0; i < count; i++)
            {
                BlockingCollection<Work> queue = new(); _commands.Add(queue);
                Thread thread = new(() =>
                {
                    foreach (Work work in queue.GetConsumingEnumerable())
                    {
                        _ready.SignalAndWait(); _start.SignalAndWait();
                        PaddleOcrResult? result = null; Exception? error = null;
                        try { for (int iteration = 0; iteration < work.Iterations; iteration++) result = pipeline.Run(work.Image.Bytes, work.Image.Width, work.Image.Height, true); }
                        catch (Exception exception) { error = exception; }
                        _finish.SignalAndWait(); _responses.Add(new(result, error));
                    }
                }) { IsBackground = true, Name = $"cpu-acceptance-{i}" };
                _threads.Add(thread); thread.Start();
            }
        }
        internal (double WallNs, Response[] Results, object? Profile) Round(Image image, int iterations, bool profile)
        {
            PipelineProfiler.Enable(profile);
            foreach (var queue in _commands) queue.Add(new(image, iterations));
            _ready.SignalAndWait(); var timer = Stopwatch.StartNew(); _start.SignalAndWait(); _finish.SignalAndWait(); double wall = timer.Elapsed.TotalNanoseconds;
            PipelineProfiler.Enable(false);
            Response[] responses = Enumerable.Range(0, _commands.Count).Select(_ => _responses.Take()).ToArray();
            foreach (Response response in responses) if (response.Error is not null) throw new InvalidOperationException("CPU acceptance worker failed", response.Error);
            object? stages = profile ? PipelineProfiler.Snapshot().Select((stage, i) => new { Stage = PipelineProfiler.StageNames[i], Nanoseconds = stage.Milliseconds * 1e6, stage.Calls }).ToArray() : null;
            return (wall, responses, stages);
        }
        public void Dispose()
        {
            foreach (var queue in _commands) queue.CompleteAdding();
            foreach (var thread in _threads) thread.Join();
            foreach (var queue in _commands) queue.Dispose();
            _responses.Dispose(); _ready.Dispose(); _start.Dispose(); _finish.Dispose();
        }
    }

    internal static void Run(string[] args)
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64) throw new PlatformNotSupportedException("CPU acceptance requires Windows x64");
        if (args.Contains("--phase-contracts")) { RunPhaseContracts(args); return; }
        if (args.Contains("--phase-only")) { RunPhaseOnly(args); return; }
        if (args.Contains("--array-pool-self-test"))
        {
            if (args.Length != 2 || !args.Contains("--acceptance")) throw new ArgumentException("ArrayPool self-test takes no model or timing arguments");
            Console.WriteLine(JsonSerializer.Serialize(NumericPoolProbe.VerifySupport(), Json)); return;
        }
        Dictionary<string, string> options = [];
        bool scratchOnly = args.Contains("--scratch-only");
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] is "--acceptance" or "--scratch-only") continue;
            if (!new[] { "--models-dir", "--corpus", "--output", "--mode", "--threads", "--warmups", "--samples", "--iterations" }.Contains(args[i]) || i + 1 >= args.Length) throw new ArgumentException("Unknown or incomplete CPU acceptance option");
            options.Add(args[i++], args[i]);
        }
        string Required(string name) => Path.GetFullPath(options.TryGetValue(name, out string? value) ? value : throw new ArgumentException("Missing " + name));
        int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString()));
        string mode = options.GetValueOrDefault("--mode", "validate"); int budget = Number("--threads", 1), warmups = Number("--warmups", 2), samples = Number("--samples", 5), iterations = Number("--iterations", 1);
        if (mode is not ("validate" or "latency" or "throughput") || budget is not (1 or 2 or 4 or 8) || warmups < 2 || samples < 5 || iterations < 1) throw new ArgumentException("Invalid fixed CPU acceptance schedule");
        int count = mode == "throughput" ? budget : 1, capacity = mode == "throughput" ? 1 : budget;
        if (Environment.ProcessorCount != capacity) throw new InvalidOperationException("DOTNET_PROCESSOR_COUNT must match the declared per-request capacity");
        string root = Required("--corpus"), models = Required("--models-dir"), output = Required("--output"); Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        using JsonDocument corpus = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "corpus.json")));
        Image[] images = corpus.RootElement.GetProperty("images").EnumerateArray().Select(image =>
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(root, image.GetProperty("file").GetString()!));
            if (Convert.ToHexStringLower(SHA256.HashData(bytes)) != image.GetProperty("sha256").GetString()) throw new InvalidDataException("Changed acceptance raw image");
            int width = image.GetProperty("width").GetInt32(), height = image.GetProperty("height").GetInt32();
            if (bytes.Length != checked(width * height * 3)) throw new InvalidDataException("Invalid acceptance image dimensions");
            return new Image(image.GetProperty("id").GetString()!, width, height, bytes);
        }).ToArray();
        if (!images.Select(x => x.Id).SequenceEqual(new[] { "sample", "dense-multilingual", "text-long" })) throw new InvalidDataException("All three fixed acceptance workloads are required");
        List<object> cases = [];
        var runtimeIdentity = new { RuntimeInformation.FrameworkDescription, EnvironmentVersion = Environment.Version.ToString(),
            ProcessArchitecture = RuntimeInformation.ProcessArchitecture.ToString(), RuntimeInformation.OSDescription,
            JitEnvironment = JitEnvironment(),
            Contract = "Inherited configuration snapshot; null means unset. This does not measure active JIT tiers, PGO completion or settled steady state." };
        object? poolSupport = mode == "validate" || scratchOnly ? NumericPoolProbe.VerifySupport() : null;
        void Save(bool complete) => File.WriteAllText(output, JsonSerializer.Serialize(new
        {
            SchemaVersion = 1, Complete = complete, Engine = "csharp", Mode = mode, Budget = budget, Workers = count, CapacityPerRequest = capacity,
            Warmups = warmups, SampleCount = samples, Iterations = iterations, OwnedOutputs = true,
            RuntimeIdentity = runtimeIdentity,
            Isa = new { Avx = Avx.IsSupported, Fma = Fma.IsSupported, Avx2 = Avx2.IsSupported, Avx512 = Avx512F.IsSupported },
            ArrayPoolProbeSupport = poolSupport,
            ScratchOnly = scratchOnly, ScratchObserverEnabled = mode == "validate" || scratchOnly,
            Cases = cases,
        }, Json));
        foreach ((string recName, string detName, string dictionary) in new[] { ("rec", "det", "ppocr_keys.txt"), ("small_rec", "small_det", "rec_keys.txt"), ("medium_rec", "medium_det", "rec_keys.txt") })
        {
            var timer = Stopwatch.StartNew(); using Model det = Load(models, detName); using Model cls = Load(models, "cls"); using Model rec = Load(models, recName); byte[] keys = File.ReadAllBytes(Path.Combine(models, dictionary)); double loadNs = timer.Elapsed.TotalNanoseconds; timer.Restart();
            using PaddleOcrAll pipeline = new(det, cls, rec, keys, new()
            {
                LineWorkerCount = Math.Min(capacity, 4), DetIntraOpThreads = capacity, PreprocessWorkerCount = capacity, RecBatchLines = 1,
                Detector = new() { Backend = OcrBackend.Cpu, LimitSideLength = 960, MaxPooledSessions = budget },
                Classifier = new() { Backend = OcrBackend.Cpu, MaxPooledSessions = budget },
                Recognizer = new() { Backend = OcrBackend.Cpu, MaxPooledSessions = budget },
            });
            using Workers workers = new(pipeline, count); double compileNs = timer.Elapsed.TotalNanoseconds;
            foreach (Image image in images)
            {
                (double WallNs, Response[] Results, object? Profile) cold = (0, [], null);
                object? poolQualification = mode == "validate" ? NumericPoolProbe.Measure(() => cold = workers.Round(image, 1, false)) : null;
                if (mode != "validate") cold = workers.Round(image, 1, !scratchOnly);
                PaddleOcrResult expected = cold.Results[0].Result!; Check(cold.Results, expected);
                Dictionary<string, object?> row = new()
                {
                    ["id"] = $"{recName}/{image.Id}", ["result"] = Result(expected), ["max_recognition_tensor_width"] = expected.Lines.Select(line => line.RecognitionTensorWidth).DefaultIfEmpty(0).Max(), ["line_count"] = expected.Lines.Length,
                    ["model_load_ns"] = loadNs, ["compile_requests_ns"] = compileNs,
                    ["cold"] = mode == "validate" || scratchOnly ? null : new { WallNs = cold.WallNs, Profile = cold.Profile, Instrumented = true, IncludesBindingAndReshape = true },
                    ["numeric_pool_scratch_qualification"] = poolQualification,
                };
                if (mode != "validate" && !scratchOnly)
                {
                    for (int i = 0; i < warmups; i++) Check(workers.Round(image, iterations, false).Results, expected);
                    var warmed = Memory(pipeline, det, cls, rec); List<object> timings = [];
                    for (int i = 0; i < samples; i++) { var batch = workers.Round(image, iterations, false); Check(batch.Results, expected); timings.Add(new { WallNs = batch.WallNs, Requests = count * iterations, BorrowedPipelineNsSum = (long?)null, OwnedCopyNsSum = (long?)null }); }
                    var after = Memory(pipeline, det, cls, rec); long allocated = GC.GetTotalAllocatedBytes(true); var probe = workers.Round(image, 1, false); allocated = GC.GetTotalAllocatedBytes(true) - allocated; Check(probe.Results, expected);
                    row["samples"] = timings; row["memory"] = new { Warmed = warmed, AfterSamples = after, RepeatedShape = Memory(pipeline, det, cls, rec), AllocationProbe = new { AllocatedBytes = allocated, AllocationCount = (long?)null, Requests = count, Scope = "one separate owned-output worker batch, including runner allocations; managed volume" }, PeakProcessWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64 };
                }
                else if (scratchOnly)
                {
                    for (int i = 0; i < warmups; i++) Check(workers.Round(image, iterations, false).Results, expected);
                    row["numeric_pool_scratch_probe"] = NumericPoolProbe.Measure(() => Check(workers.Round(image, 1, false).Results, expected));
                }
                row["shared"] = new { NormalizedWeightBytes = det.Info.WeightSize + cls.Info.WeightSize + rec.Info.WeightSize, PackedWeightBytes = Packed(det) + Packed(cls) + Packed(rec) };
                cases.Add(row); Save(false); Console.WriteLine($"acceptance csharp/{mode}/{budget}: {recName}/{image.Id}");
            }
        }
        Save(true);
    }

    private static Model Load(string root, string name) { using FileStream stream = File.OpenRead(Path.Combine(root, name + ".onnx")); return Model.Load(stream); }
    private static object Box(PaddleOcrDetectionBox box) => new { Points = new[] { new[] { box.X1, box.Y1 }, new[] { box.X2, box.Y2 }, new[] { box.X3, box.Y3 }, new[] { box.X4, box.Y4 } }, box.Score };
    private static object Result(PaddleOcrResult result) => new
    {
        result.Text, result.DetectedCount, result.DetectorResizedWidth, result.DetectorResizedHeight,
        Lines = result.Lines.Select(line => new { Detection = Box(line.Box), line.Text, line.RecognitionScore, line.ClassificationScore, line.ClassificationLabel, line.AppliedRotationDegrees, line.EmittedCount, line.RecognitionContentWidth, line.RecognitionTensorWidth, line.RecognitionTimeSteps, CtcSpans = line.CtcSpans?.Select(span => new { span.Text, span.Score, span.StartColumn, span.EndColumn }).ToArray() }).ToArray(),
    };
    private static void Check(Response[] responses, PaddleOcrResult expected)
    {
        foreach (var response in responses)
        {
            PaddleOcrResult actual = response.Result ?? throw new InvalidDataException("Missing owned result");
            if (actual.DetectedCount != expected.DetectedCount || actual.DetectorResizedWidth != expected.DetectorResizedWidth || actual.DetectorResizedHeight != expected.DetectorResizedHeight || actual.Lines.Length != expected.Lines.Length) throw new InvalidDataException("Repeated/concurrent output metadata changed");
            for (int i = 0; i < actual.Lines.Length; i++)
            {
                PaddleOcrLine a = actual.Lines[i], b = expected.Lines[i];
                if (a.Box != b.Box || a.Text != b.Text || a.RecognitionScore != b.RecognitionScore || a.ClassificationScore != b.ClassificationScore || a.ClassificationLabel != b.ClassificationLabel || a.AppliedRotationDegrees != b.AppliedRotationDegrees || a.EmittedCount != b.EmittedCount || a.RecognitionContentWidth != b.RecognitionContentWidth || a.RecognitionTensorWidth != b.RecognitionTensorWidth || a.RecognitionTimeSteps != b.RecognitionTimeSteps || (a.CtcSpans is null) != (b.CtcSpans is null) || !a.CtcSpans.AsSpan().SequenceEqual(b.CtcSpans)) throw new InvalidDataException("Repeated/concurrent line result changed");
            }
        }
    }

    private static long Packed(Model model)
    {
        var cache = (IEnumerable)(typeof(Model).GetField("_packedWeights", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(model) ?? throw new MissingFieldException("Pinned packed-weight cache changed")); long bytes = 0;
        foreach (object pair in cache)
        {
            object lazy = pair.GetType().GetProperty("Value")!.GetValue(pair)!;
            if ((bool)lazy.GetType().GetProperty("IsValueCreated")!.GetValue(lazy)! && lazy.GetType().GetProperty("Value")!.GetValue(lazy) is float[] values) bytes += values.LongLength * 4;
        }
        return bytes;
    }
    private static object Memory(PaddleOcrAll pipeline, params Model[] models)
    {
        HashSet<object> seen = new(ReferenceEqualityComparer.Instance); long activation = 0, guard = 0, arrays = 0;
        foreach (Model model in models) for (int i = 0; i < model.Tensors.Length; i++) seen.Add(model.GetTensorArray(i));
        void Visit(object? value)
        {
            if (value is null || value is string || value is Model || value is CompiledModel || !seen.Add(value)) return;
            if (value is NativeWorkspace native) { activation += native.Length * 4L; guard += NativeWorkspace.GuardFloats * 4L; return; }
            Type type = value.GetType(); if (type.IsPrimitive || type.IsEnum || type.IsPointer) return;
            if (value is Array array)
            {
                if (type.GetElementType()!.IsPrimitive) { arrays += Buffer.ByteLength(array); return; }
                long element = IntPtr.Size; if (type.GetElementType()!.IsValueType) { try { element = Marshal.SizeOf(type.GetElementType()!); } catch (ArgumentException) { } }
                arrays += array.LongLength * element; foreach (object? item in array) Visit(item); return;
            }
            if (value is IEnumerable enumerable && type.Assembly != typeof(Model).Assembly)
            {
                int count = value is ICollection collection ? collection.Count : 0;
                int capacity = type.GetProperty("Capacity")?.GetValue(value) as int? ?? count; arrays += (long)capacity * IntPtr.Size;
                foreach (object? item in enumerable) Visit(item); return;
            }
            if (type.Assembly != typeof(Model).Assembly) return;
            foreach (FieldInfo field in type.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public))
                if (!field.FieldType.IsPointer && field.Name is not ("_labels" or "_dictionary")) Visit(field.GetValue(value));
        }
        Visit(pipeline); return new { ActivationBytes = activation + guard, ActivationUsableBytes = activation, ActivationGuardBytes = guard,
            ActivationContract = "requested aligned native allocation bytes, including 64 guard floats per workspace; excludes native allocator headers/alignment bookkeeping", WorkspaceBytes = arrays,
            KernelScratchBytes = (long?)null, NonkernelWorkspaceBytes = (long?)null,
            KernelScratchContract = "unavailable: pinned CPU kernels rent transient ArrayPool buffers and use fixed stack tiles; these are not attributable to retained session fields. Null does not mean zero. Process peak working set includes committed pool/stack pages",
            WorkspaceMeasurement = "observed_payloads_plus_collection_count_estimate",
            WorkspaceContract = "observed managed array payloads plus Count-based collection-slot estimate; opaque framework collection backing capacities (including ConcurrentBag per-thread queues), object headers, transient ArrayPool rentals, fixed stack tiles and shared models/dictionary are excluded. Stable values do not prove total C# collection capacity is stable" };
    }
}
