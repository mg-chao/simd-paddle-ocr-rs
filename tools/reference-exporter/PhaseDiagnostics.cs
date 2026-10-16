// Separate diagnostic process only. Official Workers/timing never install these adapters.
using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

internal static partial class AcceptanceRunner
{
    private const string ReshapeContract = "summed-full-ocr-session-reshape-calls-including-noops-v1";
    private const string OwnedContract = "pinned-api-intrinsically-owned-no-post-run-copy-v1";
    private const string InitializationContract = "diagnostic-pools-precreated-outside-run-including-default-session-bindings-v1";

    private sealed class ReshapeCounter(Func<long>? clock = null, long frequency = 0)
    {
        private readonly Func<long> _clock = clock ?? Stopwatch.GetTimestamp;
        private readonly long _frequency = frequency == 0 ? Stopwatch.Frequency : frequency;
        private readonly long[] _ticks = new long[3], _calls = new long[3], _failures = new long[3];
        internal long Start() => _clock();
        internal void End(int stage, long started, bool failed)
        {
            long elapsed = checked(_clock() - started);
            if (elapsed < 0) throw new InvalidOperationException("Phase clock moved backward");
            Interlocked.Add(ref _ticks[stage], elapsed);
            Interlocked.Increment(ref _calls[stage]);
            if (failed) Interlocked.Increment(ref _failures[stage]);
        }
        internal void Reset() { Array.Clear(_ticks); Array.Clear(_calls); Array.Clear(_failures); }
        internal (long Nanoseconds, long Calls, long FailedCalls) Read(int stage) =>
            ((long)Math.Round(Interlocked.Read(ref _ticks[stage]) * 1e9 / _frequency),
             Interlocked.Read(ref _calls[stage]), Interlocked.Read(ref _failures[stage]));
        internal object Stages() => Enumerable.Range(0, 3).Select(stage =>
        {
            var value = Read(stage);
            return new { Stage = new[] { "det", "cls", "rec" }[stage], value.Nanoseconds, value.Calls, value.FailedCalls };
        }).ToArray();
        internal long TotalNs => Enumerable.Range(0, 3).Sum(stage => Read(stage).Nanoseconds);
    }

    private sealed class PhaseSession(IOcrSession inner, ReshapeCounter counter, int stage) : IOcrSession
    {
        public TensorShape InputShape => inner.InputShape;
        public TensorShape OutputShape => inner.OutputShape;
        public Span<float> InputData => inner.InputData;
        public ResizeWorkspace ResizeWorkspace => inner.ResizeWorkspace;
        public bool InputIsNhwc => inner.InputIsNhwc;
        public int IntraOpThreads { get => inner.IntraOpThreads; set => inner.IntraOpThreads = value; }
        public int HighWaterInputVolume => inner.HighWaterInputVolume;
        public bool PlanForCtcProjection { get => inner.PlanForCtcProjection; set => inner.PlanForCtcProjection = value; }
        public bool GpuAlive => inner.GpuAlive;
        public bool IsProfilingEnabled => inner.IsProfilingEnabled;
        public void Reshape(ReadOnlySpan<int> shape)
        {
            long started = counter.Start(); bool failed = true;
            try { inner.Reshape(shape); failed = false; }
            finally { counter.End(stage, started, failed); }
        }
        public ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input) => inner.RunInternal(input);
        public bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands) => inner.TryRunUntilCtcProjection(input, out operands);
        public ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool logits) => inner.RunInternalSkipFinalSoftmax(input, out logits);
        public void NoteProfile(OperatorId operation, long started, int nodeIndex) => inner.NoteProfile(operation, started, nodeIndex);
        public void Dispose() => inner.Dispose();
    }

    private static FieldInfo PhaseField(object owner, string name) => owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new MissingFieldException(owner.GetType().FullName, name);

    private static void AddPhaseSessions(object pool, int count, Func<IOcrSession> create, ReshapeCounter counter, int stage)
    {
        if (count < 1 || pool is not (ConcurrentBag<IOcrSession> or List<IOcrSession>)) throw new InvalidDataException("Pinned session-pool type changed");
        if (((IEnumerable)pool).Cast<object>().Any()) throw new InvalidDataException("Diagnostic pools must initially be empty");
        for (int index = 0; index < count; index++)
        {
            IOcrSession inner = create();
            // Exact flags set by pinned Classifier/Recognizer.RentSession on a new session.
            if (stage is 1 or 2) inner.PlanForCtcProjection = true;
            PhaseSession wrapped = new(inner, counter, stage);
            if (pool is ConcurrentBag<IOcrSession> bag) bag.Add(wrapped);
            else ((List<IOcrSession>)pool).Add(wrapped);
        }
    }

    private static object[] InstallPhaseSessions(PaddleOcrAll pipeline, int count, ReshapeCounter counter)
    {
        object[] pools = new object[3];
        foreach ((int stage, string field) in new[] { (0, "_detector"), (1, "_classifier"), (2, "_recognizer") })
        {
            object component = PhaseField(pipeline, field).GetValue(pipeline) ?? throw new InvalidDataException("CLS must be enabled for phase diagnostics");
            var compiled = (CompiledModel)(PhaseField(component, "_compiled").GetValue(component) ?? throw new InvalidDataException("Missing compiled CPU model"));
            object pool = PhaseField(component, "_sessions").GetValue(component) ?? throw new InvalidDataException("Missing pinned session pool");
            AddPhaseSessions(pool, count, () => OcrSessionFactory.Create(compiled, OcrBackend.Cpu), counter, stage);
            PhaseField(component, "_pooledCount").SetValue(component, count);
            pools[stage] = pool;
        }
        return pools;
    }

    private static void CheckPhasePools(object[] pools, int count)
    {
        foreach (object pool in pools)
        {
            object[] sessions = ((IEnumerable)pool).Cast<object>().ToArray();
            if (sessions.Length != count || sessions.Any(session => session is not PhaseSession))
                throw new InvalidDataException("Full OCR created, lost, or bypassed an instrumented session");
        }
    }

    private static object PhaseResult(PaddleOcrResult result, string publishedText) => new
    {
        Text = publishedText, result.DetectedCount, result.DetectorResizedWidth, result.DetectorResizedHeight,
        Lines = result.Lines.Select(line => new { Detection = Box(line.Box), line.Text, line.RecognitionScore, line.ClassificationScore,
            line.ClassificationLabel, line.AppliedRotationDegrees, line.EmittedCount, line.RecognitionContentWidth,
            line.RecognitionTensorWidth, line.RecognitionTimeSteps, CtcSpans = line.CtcSpans?.Select(span => new { span.Text, span.Score, span.StartColumn, span.EndColumn }).ToArray() }).ToArray(),
    };

    private static void RunPhaseOnly(string[] args)
    {
        Dictionary<string, string> options = [];
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] is "--acceptance" or "--phase-only") continue;
            if (!new[] { "--models-dir", "--corpus", "--output", "--mode", "--threads", "--warmups", "--samples", "--iterations", "--phase-round-id", "--phase-plan-sha256" }.Contains(args[i]) || i + 1 >= args.Length)
                throw new ArgumentException("Unknown or incomplete phase diagnostic option");
            options.Add(args[i++], args[i]);
        }
        string Required(string key) => options.TryGetValue(key, out string? value) ? value : throw new ArgumentException("Missing " + key);
        string roundId = Required("--phase-round-id"), planSha = Required("--phase-plan-sha256");
        if (!Guid.TryParseExact(roundId, "D", out _) || planSha.Length != 64 || planSha.Any(c => !Uri.IsHexDigit(c)))
            throw new ArgumentException("Phase diagnostics require a reserved-round UUID and frozen-plan SHA256");
        string mode = Required("--mode"); int budget = int.Parse(Required("--threads"));
        if (mode is not ("latency" or "throughput") || budget is not (1 or 2 or 4 or 8)) throw new ArgumentException("Phase diagnostics must match an official paired partition");
        int workers = mode == "throughput" ? budget : 1, capacity = mode == "throughput" ? 1 : budget;
        if (Environment.ProcessorCount != capacity) throw new InvalidOperationException("Diagnostic DOTNET_PROCESSOR_COUNT changed the CPU partition");
        string root = Path.GetFullPath(Required("--corpus")), models = Path.GetFullPath(Required("--models-dir")), output = Path.GetFullPath(Required("--output"));
        if (File.Exists(output)) throw new InvalidOperationException("Preserve the existing phase diagnostic output");
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        using JsonDocument corpus = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "corpus.json")));
        Image[] images = corpus.RootElement.GetProperty("images").EnumerateArray().Select(item =>
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(root, item.GetProperty("file").GetString()!));
            int width = item.GetProperty("width").GetInt32(), height = item.GetProperty("height").GetInt32();
            if (width < 1 || height < 1 || bytes.Length != checked(width * height * 3)
                || Convert.ToHexStringLower(SHA256.HashData(bytes)) != item.GetProperty("sha256").GetString()) throw new InvalidDataException("Changed phase raw corpus");
            return new Image(item.GetProperty("id").GetString()!, width, height, bytes);
        }).ToArray();
        if (!images.Select(image => image.Id).SequenceEqual(new[] { "sample", "dense-multilingual", "text-long" })) throw new InvalidDataException("Phase workloads changed");
        List<object> cases = [];
        void Save(bool complete) => File.WriteAllText(output, JsonSerializer.Serialize(new
        {
            SchemaVersion = 1, Complete = complete, Engine = "csharp", PhaseOnly = true, OfficialSamples = false,
            Mode = mode, Budget = budget, Workers = workers, CapacityPerRequest = capacity,
            PhaseRoundId = roundId, PhasePlanSha256 = planSha,
            RuntimeIdentity = new { RuntimeInformation.FrameworkDescription, EnvironmentVersion = Environment.Version.ToString(),
                ProcessArchitecture = RuntimeInformation.ProcessArchitecture.ToString(), RuntimeInformation.OSDescription,
                JitEnvironment = JitEnvironment(), Contract = "Inherited runtime configuration, not settled JIT tier proof" },
            ReshapeContract, InitializationContract, Cases = cases,
        }, Json));
        foreach ((string recName, string detName, string dictionary) in new[] { ("rec", "det", "ppocr_keys.txt"), ("small_rec", "small_det", "rec_keys.txt"), ("medium_rec", "medium_det", "rec_keys.txt") })
        {
            var timer = Stopwatch.StartNew(); using Model det = Load(models, detName); using Model cls = Load(models, "cls"); using Model rec = Load(models, recName);
            byte[] keys = File.ReadAllBytes(Path.Combine(models, dictionary)); double loadNs = timer.Elapsed.TotalNanoseconds; timer.Restart();
            using PaddleOcrAll pipeline = new(det, cls, rec, keys, new()
            {
                LineWorkerCount = Math.Min(capacity, 4), DetIntraOpThreads = capacity, PreprocessWorkerCount = capacity, RecBatchLines = 1,
                Detector = new() { Backend = OcrBackend.Cpu, LimitSideLength = 960, MaxPooledSessions = budget },
                Classifier = new() { Backend = OcrBackend.Cpu, MaxPooledSessions = budget },
                Recognizer = new() { Backend = OcrBackend.Cpu, MaxPooledSessions = budget },
            });
            ReshapeCounter counter = new(); var creation = Stopwatch.StartNew(); object[] pools = InstallPhaseSessions(pipeline, budget, counter);
            double creationNs = creation.Elapsed.TotalNanoseconds;
            using Workers pool = new(pipeline, workers); double compileNs = timer.Elapsed.TotalNanoseconds;
            foreach (Image image in images)
            {
                counter.Reset(); var run = pool.Round(image, 1, true);
                PaddleOcrResult expected = run.Results[0].Result ?? throw new InvalidDataException("Missing phase owned result"); Check(run.Results, expected);
                CheckPhasePools(pools, budget);
                long lineCalls = checked((long)workers * expected.Lines.Length);
                if (counter.Read(0).Calls != workers || counter.Read(1).Calls != lineCalls || counter.Read(2).Calls != lineCalls
                    || Enumerable.Range(0, 3).Any(stage => counter.Read(stage).FailedCalls != 0)) throw new InvalidDataException("Full OCR reshape coverage differs from actual DET/CLS/REC requests");
                double publicationNs = 0; string? published = null;
                foreach (Response response in run.Results)
                {
                    timer.Restart(); string text = response.Result!.Text; publicationNs += timer.Elapsed.TotalNanoseconds;
                    if (published is not null && published != text) throw new InvalidDataException("Phase text publication changed concurrent output");
                    published = text;
                }
                cases.Add(new
                {
                    Id = $"{recName}/{image.Id}", Result = PhaseResult(expected, published!), LineCount = expected.Lines.Length,
                    ModelLoadNs = loadNs, CompileRequestsNs = compileNs,
                    Phase = new { FirstRunWallNs = run.WallNs, IncludesReshapeAndInstrumentation = true,
                        ColdContract = "first-run-of-this-input-after-explicit-diagnostic-session-creation; earlier-shape-caches-may-exist",
                        DiagnosticSessionCreationNs = creationNs, DiagnosticSessionCreationCalls = budget * 3,
                        ReshapeNs = counter.TotalNs, ReshapeStages = counter.Stages(),
                        ReshapeTimeKind = "work-sum; may-overlap-across-workers; never-subtract-from-wall-time",
                        PinnedPipelineStages = run.Profile,
                        PostRunOwnedCopyNs = 0L, OwnershipContract = OwnedContract,
                        OwnershipExplanation = "Pinned Run allocates owned line objects, strings, CTC span arrays and result container inside pipeline time. It has no borrowed result or external owned-copy operation.",
                        PostRunTextPublicationNs = publicationNs,
                        TextPublicationContract = "actual-PaddleOcrResult.Text-string.Join-once-per-worker-after-Run; separate-diagnostic-work-sum",
                        OfficialTimingChanged = false,
                    },
                });
                Save(false);
            }
        }
        Save(true);
    }

    private sealed class FakePhaseSession : IOcrSession
    {
        internal Exception? NextFailure; internal int ReshapeCalls, RunCalls, DisposeCalls, ProfileCalls;
        private readonly float[] _values = [1, 2, 3];
        public TensorShape InputShape => new([1, 3]); public TensorShape OutputShape => new([1, 3]);
        public Span<float> InputData => _values; public ResizeWorkspace ResizeWorkspace { get; } = new();
        public bool InputIsNhwc => true; public int IntraOpThreads { get; set; } = 2;
        public int HighWaterInputVolume => 3; public bool PlanForCtcProjection { get; set; }
        public bool GpuAlive => false; public bool IsProfilingEnabled => false;
        public void Reshape(ReadOnlySpan<int> shape) { ReshapeCalls++; if (NextFailure is Exception failure) { NextFailure = null; throw failure; } }
        public ReadOnlySpan<float> RunInternal(ReadOnlySpan<float> input) { RunCalls++; return _values; }
        public bool TryRunUntilCtcProjection(ReadOnlySpan<float> input, out CtcProjectionOperands operands) { RunCalls++; operands = default; return false; }
        public ReadOnlySpan<float> RunInternalSkipFinalSoftmax(ReadOnlySpan<float> input, out bool logits) { RunCalls++; logits = true; return _values; }
        public void NoteProfile(OperatorId operation, long started, int nodeIndex) => ProfileCalls++;
        public void Dispose() => DisposeCalls++;
    }

    private static void RunPhaseContracts(string[] args)
    {
        if (!args.SequenceEqual(new[] { "--acceptance", "--phase-contracts" })) throw new ArgumentException("Fake phase contracts take no model or timing arguments");
        static void Assert(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
        long now = 0; ReshapeCounter counter = new(() => Interlocked.Add(ref now, 7), 1_000_000_000);
        FakePhaseSession inner = new(); PhaseSession wrapped = new(inner, counter, 0);
        wrapped.Reshape([1, 3]); Assert(inner.ReshapeCalls == 1 && counter.Read(0) == (7, 1, 0), "Independent fake-clock reshape cost/count");
        wrapped.InputData[0] = 9; Assert(inner.InputData[0] == 9 && wrapped.InputIsNhwc && wrapped.HighWaterInputVolume == 3, "Borrowed input/state forwarding");
        wrapped.IntraOpThreads = 4; wrapped.PlanForCtcProjection = true;
        Assert(inner.IntraOpThreads == 4 && inner.PlanForCtcProjection, "Mutable execution/CTC flag forwarding");
        Assert(wrapped.RunInternal([])[0] == 9 && !wrapped.TryRunUntilCtcProjection([], out _)
            && wrapped.RunInternalSkipFinalSoftmax([], out bool logits)[0] == 9 && logits && inner.RunCalls == 3, "All graph/head routes forward unchanged");
        Exception failure = new InvalidDataException("injected reshape"); inner.NextFailure = failure;
        try { wrapped.Reshape([2, 3]); throw new InvalidOperationException("Missing original reshape failure"); }
        catch (InvalidDataException actual) { Assert(ReferenceEquals(actual, failure), "Original failure identity preserved"); }
        Assert(counter.Read(0) == (14, 2, 1), "Failed reshape cost/count retained");
        counter.Reset(); Parallel.For(0, 256, _ => wrapped.Reshape([1, 3]));
        Assert(counter.Read(0).Calls == 256 && counter.Read(0).FailedCalls == 0 && counter.TotalNs >= 256 * 7, "Concurrent work counters");
        wrapped.NoteProfile(default, 0, 0); wrapped.Dispose(); Assert(inner.ProfileCalls == 1 && inner.DisposeCalls == 1, "Profile/dispose forwarding");
        foreach (object pool in new object[] { new ConcurrentBag<IOcrSession>(), new List<IOcrSession>() })
        {
            AddPhaseSessions(pool, 4, () => new FakePhaseSession(), counter, 2);
            CheckPhasePools([pool], 4);
            Assert(((IEnumerable)pool).Cast<IOcrSession>().All(session => session.PlanForCtcProjection), "Pinned new REC session flag preserved");
            try { AddPhaseSessions(pool, 4, () => new FakePhaseSession(), counter, 2); throw new InvalidOperationException("Nonempty pool accepted"); }
            catch (InvalidDataException) { }
            foreach (IOcrSession session in (IEnumerable)pool) session.Dispose();
        }
        try { CheckPhasePools([new List<IOcrSession> { new FakePhaseSession() }], 1); throw new InvalidOperationException("Unwrapped bypass accepted"); }
        catch (InvalidDataException) { }
        Console.WriteLine(JsonSerializer.Serialize(new { Passed = true, FakeSessionsOnly = true, RealModelsRead = false,
            PerformanceRoundReserved = false, Clock = "injected-exact-seven-ticks-per-pair", Contracts = 10 }, Json));
    }
}
