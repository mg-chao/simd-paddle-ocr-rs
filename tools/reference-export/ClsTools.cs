using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Graph-only timing of the verified, frozen classifier corpus.</summary>
internal static class ClsTools
{
    private sealed record Input(string CaseId, string ArtifactId, int[] Shape, string Sha256, float[] Values);
    private sealed record Corpus(string ModelPath, string ModelSha256, string ManifestSha256, Input[] Inputs);

    public static int Benchmark(string[] args)
    {
        Dictionary<string, string> options = Parse(args);
        string root = Path.GetFullPath(Required(options, "reference"));
        int warmup = Count(options, "warmup", 30), samples = Count(options, "samples", 100);
        if (warmup < 0 || samples < 1) throw new ArgumentException("Invalid benchmark counts.");
        RequireCanonical();
        Corpus corpus = ReadCorpus(root);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        long started = Stopwatch.GetTimestamp();
        using FileStream modelStream = File.OpenRead(corpus.ModelPath);
        using Model model = Model.Load(modelStream);
        double loadMs = Elapsed(started);
        started = Stopwatch.GetTimestamp();
        List<InferenceSession> sessions = [];
        List<int[]> shapes = [];
        int[] sessionIndexes = new int[corpus.Inputs.Length];
        try
        {
            for (int i = 0; i < corpus.Inputs.Length; i++)
            {
                int[] shape = corpus.Inputs[i].Shape;
                int index = shapes.FindIndex(existing => existing.AsSpan().SequenceEqual(shape));
                if (index < 0)
                {
                    index = sessions.Count;
                    sessions.Add(new InferenceSession(model, shape, 1));
                    shapes.Add(shape);
                }
                sessionIndexes[i] = index;
            }
            double prepareMs = Elapsed(started);
            double checksum = 0;
            double[] firstCalls = new double[corpus.Inputs.Length];
            long firstSweep = Stopwatch.GetTimestamp();
            for (int i = 0; i < corpus.Inputs.Length; i++)
            {
                started = Stopwatch.GetTimestamp();
                ReadOnlySpan<float> prediction = sessions[sessionIndexes[i]].RunInternal(corpus.Inputs[i].Values);
                firstCalls[i] = Elapsed(started);
                checksum += prediction[0];
            }
            double firstSweepMs = Elapsed(firstSweep);
            for (int sweep = 0; sweep < warmup; sweep++)
                for (int i = 0; i < corpus.Inputs.Length; i++)
                    checksum += sessions[sessionIndexes[i]].RunInternal(corpus.Inputs[i].Values)[0];

            // Allocate bookkeeping before measuring managed allocation. Timing uses timestamp
            // values instead of creating Stopwatch objects or materializing graph outputs.
            double[][] times = corpus.Inputs.Select(_ => new double[samples]).ToArray();
            double[] sweepTimes = new double[samples], wallTimes = new double[samples];
            long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int sweep = 0; sweep < samples; sweep++)
            {
                long wallStarted = Stopwatch.GetTimestamp();
                double graphMs = 0;
                for (int i = 0; i < corpus.Inputs.Length; i++)
                {
                    started = Stopwatch.GetTimestamp();
                    ReadOnlySpan<float> prediction = sessions[sessionIndexes[i]].RunInternal(corpus.Inputs[i].Values);
                    double milliseconds = Elapsed(started);
                    times[i][sweep] = milliseconds;
                    graphMs += milliseconds;
                    checksum += prediction[0];
                }
                wallTimes[sweep] = Elapsed(wallStarted);
                sweepTimes[sweep] = graphMs;
            }
            long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
            using Process process = Process.GetCurrentProcess();
            process.Refresh();
            Console.WriteLine(Encoding.UTF8.GetString(BundleWriter.SerializeJson(new
            {
                schemaVersion = 1, runtime = "csharp", profile = "canonical", reference = root,
                framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
                corpus.ModelSha256, corpus.ManifestSha256, warmup, samples,
                loadMs, prepareMs, firstSweepMs, distinctShapes = sessions.Count,
                activationBytes = (long?)null, naiveActivationBytes = (long?)null,
                instrumentation = "Uninstrumented graph latency; current-thread managed allocation count is sampled outside the timed graph calls. Native allocation is not measured.",
                warmAllocationCount = (long?)null, warmAllocationBytes = allocatedBytes,
                workingSetBytes = process.WorkingSet64, peakWorkingSetBytes = process.PeakWorkingSet64,
                measurement = "Complete manifest-order corpus sweeps. Graph call includes session input binding and borrowed output; tracing, hashing, file I/O and output copying are excluded. Corpus samples sum individual graph times; sweep wall samples include timestamp/bookkeeping overhead. First calls share prepared sessions; only the first call is process-cold.",
                checksum,
                inputs = corpus.Inputs.Select((input, i) => new
                {
                    input.CaseId, input.ArtifactId, input.Shape, input.Sha256,
                    firstCallMs = firstCalls[i], samplesMs = times[i],
                    p50Ms = Percentile(times[i], .5), p95Ms = Percentile(times[i], .95),
                    meanMs = times[i].Average(), inputsPerSecond = 1000 / times[i].Average()
                }).ToArray(),
                corpus = new
                {
                    inputs = corpus.Inputs.Length, samplesMs = sweepTimes, sweepWallSamplesMs = wallTimes,
                    p50Ms = Percentile(sweepTimes, .5), p95Ms = Percentile(sweepTimes, .95),
                    meanMs = sweepTimes.Average(), inputsPerSecond = 1000 * corpus.Inputs.Length / sweepTimes.Average(),
                    wallInputsPerSecond = 1000 * corpus.Inputs.Length / wallTimes.Average()
                }
            })));
            return 0;
        }
        finally { foreach (InferenceSession session in sessions) session.Dispose(); }
    }

    private static Corpus ReadCorpus(string root)
    {
        byte[] manifestBytes = File.ReadAllBytes(Path.Combine(root, "manifest.json"));
        using JsonDocument manifest = JsonDocument.Parse(manifestBytes);
        JsonElement document = manifest.RootElement, contract = document.GetProperty("contract");
        if (document.GetProperty("schema_version").GetInt32() != 1 ||
            contract.GetProperty("profile").GetString() != "canonical" ||
            contract.GetProperty("backend").GetString() != "cpu" ||
            contract.GetProperty("logical_input_layout").GetString() != "nchw")
            throw new InvalidDataException("CLS benchmark requires a canonical CPU NCHW reference bundle.");
        string? modelPath = null;
        string? modelHash = null;
        foreach (JsonElement asset in document.GetProperty("models").EnumerateArray())
        {
            VerifyBundleFile(root, asset);
            if (asset.GetProperty("id").GetString() != "cls") continue;
            if (modelPath is not null) throw new InvalidDataException("Duplicate CLS model.");
            modelPath = BundleWriter.DestinationPath(root, asset.GetProperty("path").GetString()!);
            modelHash = asset.GetProperty("sha256").GetString();
        }
        if (modelPath is null || modelHash is null) throw new InvalidDataException("Missing CLS model.");
        if (document.TryGetProperty("auxiliary_files", out JsonElement auxiliary))
            foreach (JsonElement file in auxiliary.EnumerateArray()) VerifyBundleFile(root, file);
        List<Input> inputs = [];
        HashSet<string> caseIds = new(StringComparer.Ordinal);
        foreach (JsonElement item in document.GetProperty("cases").EnumerateArray())
        {
            string caseId = item.GetProperty("id").GetString()!;
            if (!caseIds.Add(caseId)) throw new InvalidDataException("Duplicate case ID.");
            HashSet<string> artifactIds = new(StringComparer.Ordinal);
            foreach (JsonElement artifact in item.GetProperty("artifacts").EnumerateArray())
            {
                string id = artifact.GetProperty("id").GetString()!;
                if (!artifactIds.Add(id)) throw new InvalidDataException("Duplicate artifact ID.");
                if (!IsClsInput(id)) { VerifyBundleFile(root, artifact); continue; }
                byte[] bytes = Program.ReadBundleFile(root, artifact);
                int[] shape = artifact.GetProperty("shape").EnumerateArray().Select(dimension => dimension.GetInt32()).ToArray();
                if (artifact.GetProperty("dtype").GetString() != "f32" || artifact.GetProperty("layout").GetString() != "nchw" ||
                    shape.Length != 4 || shape[0] <= 0 || shape[1] != 3 || shape[2] != 80 || shape[3] != 160 ||
                    checked((long)shape[0] * 3 * 80 * 160 * 4) != bytes.LongLength)
                    throw new InvalidDataException($"Invalid CLS input {caseId}/{id}.");
                float[] values = new float[bytes.Length / 4];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
                    if (!float.IsFinite(values[i])) throw new InvalidDataException($"Non-finite CLS input {caseId}/{id}.");
                }
                inputs.Add(new(caseId, id, shape, artifact.GetProperty("sha256").GetString()!, values));
            }
        }
        if (inputs.Count == 0) throw new InvalidDataException("Reference contains no captured CLS inputs.");
        return new(modelPath, modelHash, Convert.ToHexStringLower(SHA256.HashData(manifestBytes)), inputs.ToArray());
    }

    private static bool IsClsInput(string id)
    {
        if (id.Length < 18 || !id.StartsWith("line.", StringComparison.Ordinal) || !id.EndsWith(".cls.input", StringComparison.Ordinal)) return false;
        ReadOnlySpan<char> index = id.AsSpan(5, id.Length - 15);
        return index.Length >= 3 && index.IndexOfAnyExceptInRange('0', '9') < 0;
    }
    private static void VerifyBundleFile(string root, JsonElement descriptor)
    {
        string relative = descriptor.GetProperty("path").GetString()!;
        using FileStream stream = File.OpenRead(BundleWriter.DestinationPath(root, relative));
        long expected = descriptor.GetProperty("bytes").GetInt64();
        if (expected < 0 || stream.Length != expected) throw new InvalidDataException($"Invalid length for {relative}.");
        bool floats = descriptor.TryGetProperty("dtype", out JsonElement dtype) && dtype.GetString() == "f32";
        if (floats && expected % 4 != 0) throw new InvalidDataException($"Invalid F32 length for {relative}.");
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> buffer = stackalloc byte[64 * 1024];
        long remaining = expected;
        while (remaining > 0)
        {
            int count = (int)Math.Min(remaining, buffer.Length);
            stream.ReadExactly(buffer[..count]);
            hash.AppendData(buffer[..count]);
            if (floats)
                for (int offset = 0; offset < count; offset += 4)
                    if (!float.IsFinite(BinaryPrimitives.ReadSingleLittleEndian(buffer.Slice(offset, 4))))
                        throw new InvalidDataException($"Non-finite F32 payload in {relative}.");
            remaining -= count;
        }
        if (!Convert.ToHexStringLower(hash.GetHashAndReset()).Equals(descriptor.GetProperty("sha256").GetString(), StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Corrupted reference payload: {relative}.");
    }
    private static void RequireCanonical()
    {
        if (Environment.ProcessorCount != 1 || Vector.IsHardwareAccelerated ||
            Environment.GetEnvironmentVariable("DOTNET_PROCESSOR_COUNT") != "1" ||
            Environment.GetEnvironmentVariable("DOTNET_EnableHWIntrinsic") != "0" ||
            Environment.GetEnvironmentVariable("PPOCR_NHWC") != "0" ||
            Environment.GetEnvironmentVariable("SIMD_OCR_NOFUSE") != "1")
            throw new InvalidOperationException("Use benchmark-cls.ps1 to set canonical controls before process startup.");
        foreach (string name in new[] { "PPOCR_DUMP_CONV", "PPOCR_DUMP_NODES", "PPOCR_DUMP_PLAN", "PPOCR_DUMP_GRAPH" })
            if (Environment.GetEnvironmentVariable(name) is not null)
                throw new InvalidOperationException($"{name} must be absent from untraced CLS timing.");
    }
    private static double Elapsed(long started) => (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
    private static double Percentile(double[] samples, double fraction)
    {
        double[] sorted = (double[])samples.Clone();
        Array.Sort(sorted);
        double position = (sorted.Length - 1) * fraction;
        int left = (int)Math.Floor(position), right = (int)Math.Ceiling(position);
        return sorted[left] + (sorted[right] - sorted[left]) * (position - left);
    }
    private static Dictionary<string, string> Parse(string[] args)
    {
        Dictionary<string, string> options = new(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
            if (i + 1 >= args.Length || args[i] is not ("--reference" or "--warmup" or "--samples") ||
                !options.TryAdd(args[i][2..], args[i + 1]))
                throw new ArgumentException("Use --benchmark-cls --reference <bundle> [--warmup 30] [--samples 100].");
        return options;
    }
    private static string Required(Dictionary<string, string> options, string key) => options.TryGetValue(key, out string? value)
        ? value : throw new ArgumentException("Missing --" + key);
    private static int Count(Dictionary<string, string> options, string key, int fallback) => options.TryGetValue(key, out string? value)
        ? int.Parse(value, CultureInfo.InvariantCulture) : fallback;
}
