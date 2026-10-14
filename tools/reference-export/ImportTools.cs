using System.Diagnostics;
using System.Text;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace SimdPaddleOCR.ReferenceExport;

internal sealed record AuxiliaryFile(string Id, string Path, string Sha256, long Bytes, string Role);

internal static class ImportTools
{
    private static readonly string[] ModelNames = ["cls", "det", "rec", "small_det", "small_rec", "medium_det", "medium_rec"];

    internal static AuxiliaryFile Auxiliary(Artifact file) => new(file.Id, file.Path, file.Sha256, file.Bytes, "diagnostic");
    internal static AuxiliaryFile[] ProvenanceFiles(ReferenceProvenance provenance) => provenance.SourceInputs.Files
        .Concat(provenance.ExporterInputs.Files)
        .Concat(provenance.Git.TrackedDiff is null ? [] : new[] { provenance.Git.TrackedDiff })
        .Concat(provenance.Launcher is null ? [] : new[] { provenance.Launcher })
        .Select(file => new AuxiliaryFile(file.Path, file.Path, file.Sha256, file.Bytes, "provenance"))
        .OrderBy(file => file.Path, StringComparer.Ordinal).ToArray();

    public static int Export(string[] args)
    {
        Dictionary<string, string> options = Options(args);
        string repo = Required(options, "reference-repo");
        string exporter = Required(options, "exporter-source");
        BundleWriter writer = new(Required(options, "out"));
        List<ModelAsset> assets = [];
        List<object> contracts = [];
        List<AuxiliaryFile> auxiliary = [];
        foreach (string name in ModelNames)
        {
            ModelAsset asset = writer.CopyModel(name, Path.Combine(repo, "models", name + ".onnx"), "models/" + name + ".onnx");
            assets.Add(asset);
            AuxiliaryFile? raw = null, normalized = null;
            using FileStream input = File.OpenRead(Path.Combine(writer.RootPath, asset.Path));
            using Model model = Model.Load(input, snapshot =>
            {
                int ordinal = 0;
                object ProjectTensor(ModelInspectionTensor tensor, bool persist)
                {
                    string? path = null, hash = null;
                    if (tensor.Constant)
                    {
                        hash = BundleWriter.Hash(tensor.Data.Span);
                        if (persist)
                        {
                            path = $"weights/{name}/{ordinal++:D4}.bin";
                            BundleWriter.WriteNewFile(writer.RootPath, path, tensor.Data.Span);
                            auxiliary.Add(new("weight:" + name + ":" + tensor.Key, path, hash, tensor.Data.Length, "normalized_weight"));
                        }
                    }
                    return new { tensor.Key, tensor.Name, tensor.Dtype, tensor.Shape, tensor.Constant,
                        dataPath = path, dataBytes = tensor.Data.Length, dataSha256 = hash };
                }
                object ProjectAttribute(ModelInspectionAttribute attribute) => new
                {
                    attribute.Kind, value = attribute.Value is ModelInspectionTensor tensor
                        ? ProjectTensor(tensor, snapshot.Stage == "normalized") : attribute.Value,
                    attribute.Values, attribute.Bits, attribute.Hex
                };
                object ProjectNode(ModelInspectionNode n) => new
                {
                    n.Key, n.Name, n.OpType, n.Domain, n.Opset, n.SourceIndices, n.Inputs, n.Outputs,
                    attributes = n.Attributes.ToDictionary(pair => pair.Key, pair => ProjectAttribute(pair.Value), StringComparer.Ordinal)
                };
                object contract = new
                {
                    snapshot.SchemaVersion, snapshot.Stage, snapshot.IrVersion, snapshot.Opsets,
                    snapshot.Inputs, snapshot.Outputs,
                    tensors = snapshot.Tensors.Select(t => ProjectTensor(t, snapshot.Stage == "normalized")).ToArray(),
                    nodes = snapshot.Nodes.Select(ProjectNode).ToArray(),
                    shapeMetadata = new
                    {
                        nodes = snapshot.ShapeMetadata.Nodes.Select(ProjectNode).ToArray(),
                        snapshot.ShapeMetadata.Constants
                    },
                    snapshot.Recipes
                };
                AuxiliaryFile file = Auxiliary(writer.WriteJson(name + "." + snapshot.Stage,
                    $"contracts/{name}/{snapshot.Stage}.json", contract));
                if (snapshot.Stage == "raw") raw = file; else normalized = file;
            });
            if (raw is null || normalized is null) throw new InvalidOperationException("Importer did not deliver both inspection stages.");
            contracts.Add(new { modelId = name, raw, normalized });
            Console.WriteLine($"captured import {name}: raw + normalized before layout planning");
        }
        List<object> dictionaries = [];
        foreach (string name in new[] { "ppocr_keys", "rec_keys" })
        {
            ModelAsset file = writer.CopyModel(name, Path.Combine(repo, "models", name + ".txt"), "models/" + name + ".txt");
            string text = new UTF8Encoding(false, true).GetString(File.ReadAllBytes(Path.Combine(writer.RootPath, file.Path)));
            if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
            string[] labels = text.Split('\n');
            if (labels.Length > 0 && labels[^1].Length == 0) labels = labels[..^1];
            labels = labels.Select(label => label.EndsWith('\r') ? label[..^1] : label).ToArray();
            if (labels.Length == 0 || labels.All(label => label.Length == 0)) throw new InvalidDataException("Dictionary is empty.");
            dictionaries.Add(new { file.Id, file.Path, file.Sha256, file.Bytes, labels = labels.Length,
                classCount = labels.Length + 2, blankClass = 0, spaceClass = labels.Length + 1 });
        }
        ReferenceProvenance provenance = (ReferenceProvenance)Provenance.Capture(repo, exporter, writer.RootPath, "export-model-contracts.ps1");
        auxiliary.AddRange(ProvenanceFiles(provenance));
        writer.WriteManifest(new
        {
            schemaVersion = 1, kind = "model_import_contracts", models = assets, dictionaries, contracts,
            auxiliaryFiles = auxiliary.OrderBy(file => file.Path, StringComparer.Ordinal).ToArray(), provenance,
            contract = new { stage = "before_layout", tensorEncoding = "little_endian", generatedNames = "diagnostic",
                lineageKeys = "original:<name>; generated:bn_weight/bn_bias:<conv-index>:<bn-index>; generated:branch_weight/branch_bias:<add-index>",
                callbackLifetime = "Synchronous snapshots; raw before rewriting, normalized after pruning and before physical layout planning." }
        });
        Console.WriteLine($"saved import contracts: {writer.RootPath}");
        return 0;
    }

    public static int Benchmark(string[] args)
    {
        Dictionary<string, string> options = Options(args);
        string repo = Required(options, "reference-repo");
        string name = Required(options, "model");
        if (name != "all" && !ModelNames.Contains(name)) throw new ArgumentException("Unknown model name.");
        int warmup = Integer(options, "warmup", 3), samples = Integer(options, "samples", 30);
        if (warmup < 0 || samples < 1) throw new ArgumentException("Invalid benchmark counts.");
        if (Environment.GetEnvironmentVariable("PPOCR_NHWC") != "0")
            throw new InvalidOperationException("Import benchmark requires PPOCR_NHWC=0 before process startup.");
        string[] names = name == "all" ? ModelNames : [name];
        // Match the production path loader: stream open and page-warm reads are
        // inside timing. One sample retains every sequential model until stop.
        string[] paths = names.Select(n => Path.Combine(repo, "models", n + ".onnx")).ToArray();
        for (int i = 0; i < warmup; i++) Load(paths, out _, out _, out _);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long baselineWorkingSet = WorkingSet();
        List<object> measurements = [];
        List<double> durations = [];
        for (int i = 0; i < samples; i++)
        {
            double milliseconds = Load(paths, out long allocated, out long resident, out long peak);
            durations.Add(milliseconds);
            measurements.Add(new { index = i, durationMs = milliseconds, allocatedBytes = allocated,
                loadedWorkingSetBytes = resident, processPeakWorkingSetBytes = peak });
        }
        double[] sorted = durations.Order().ToArray();
        string output = Path.GetFullPath(Required(options, "out"));
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        using FileStream file = new(output, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        file.Write(BundleWriter.SerializeJson(new
        {
            schemaVersion = 1, runtime = "csharp", name, names, warmup, samples, baselineWorkingSetBytes = baselineWorkingSet,
            sourceFileBytes = paths.Sum(p => new FileInfo(p).Length), p50Ms = Percentile(sorted, .5), p95Ms = Percentile(sorted, .95),
            measurement = "Model.Load(FileStream), including stream open and page-warm file reads; no inspection, hashing, export or archive I/O in timed region. All sequential models retained until stop; disposal follows timing.",
            framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            nhwc = Environment.GetEnvironmentVariable("PPOCR_NHWC"),
            processPeakSemantics = "Process high-water resident memory includes warmup; baseline recorded after warmup GC. allocated_bytes counts total managed allocation volume, not a heap peak.",
            measurements
        }));
        Console.WriteLine($"import {name}: p50={Percentile(sorted, .5):F3}ms p95={Percentile(sorted, .95):F3}ms ({samples} samples)");
        return 0;
    }

    private static double Load(string[] paths, out long allocated, out long resident, out long peak)
    {
        List<Model> loaded = [];
        long before = GC.GetTotalAllocatedBytes(precise: true);
        Stopwatch clock = Stopwatch.StartNew();
        try
        {
            foreach (string path in paths)
            {
                using FileStream stream = File.OpenRead(path);
                loaded.Add(Model.Load(stream));
            }
            clock.Stop();
            allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
            using Process process = Process.GetCurrentProcess();
            process.Refresh(); resident = process.WorkingSet64; peak = process.PeakWorkingSet64;
            return clock.Elapsed.TotalMilliseconds;
        }
        finally { foreach (Model model in loaded) model.Dispose(); }
    }
    private static long WorkingSet() { using Process process = Process.GetCurrentProcess(); process.Refresh(); return process.WorkingSet64; }
    private static double Percentile(double[] sorted, double fraction)
    {
        double index = (sorted.Length - 1) * fraction; int left = (int)Math.Floor(index), right = (int)Math.Ceiling(index);
        return sorted[left] + (sorted[right] - sorted[left]) * (index - left);
    }
    private static Dictionary<string, string> Options(string[] args)
    {
        Dictionary<string, string> result = new(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
                throw new ArgumentException("Expected --name value options.");
            if (!result.TryAdd(args[i][2..], args[i + 1])) throw new ArgumentException("Duplicate option: " + args[i]);
        }
        return result;
    }
    private static string Required(Dictionary<string, string> options, string key) => options.TryGetValue(key, out string? value)
        ? value : throw new ArgumentException("Missing --" + key);
    private static int Integer(Dictionary<string, string> options, string key, int fallback) => options.TryGetValue(key, out string? value)
        ? int.Parse(value, System.Globalization.CultureInfo.InvariantCulture) : fallback;
}
