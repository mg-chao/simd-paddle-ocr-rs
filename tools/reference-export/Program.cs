using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SimdPaddleOCR.ReferenceExport;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 0 || args.Contains("--help"))
            {
                Console.WriteLine("Use tools/export-reference.ps1 -Out <new-directory> [-InputPath <image-or-directory> | -ReplayBundle <frozen-bundle>] [-Profile canonical|cpu] [-Nodes checkpoints|all] [-MaxTracedLines 3] [-Count 20] [-Warmup 1] [-Repeat 3].");
                return args.Length == 0 ? 2 : 0;
            }
            Export(Settings.Parse(args));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"reference export failed: {exception.Message}");
            return 2;
        }
    }

    private static void Export(Settings settings)
    {
        // Runtime CPU detection and ProcessorCount are initialized at process startup.
        // The launcher sets these controls before this process starts.
        if (Environment.ProcessorCount != 1)
            throw new InvalidOperationException("Start through export-reference.ps1: DOTNET_PROCESSOR_COUNT must be 1 before process startup.");
        RequireEnvironment("PPOCR_NHWC", settings.Profile == "canonical" ? "0" : "1");
        RequireEnvironment("SIMD_OCR_NOFUSE", settings.Profile == "canonical" ? "1" : "0");
        RequireEnvironment("DOTNET_EnableHWIntrinsic", settings.Profile == "canonical" ? "0" : "1");
        foreach (string name in new[] { "PPOCR_DUMP_CONV", "PPOCR_DUMP_NODES", "PPOCR_DUMP_PLAN", "PPOCR_DUMP_GRAPH" })
            if (Environment.GetEnvironmentVariable(name) is not null)
                throw new InvalidOperationException($"{name} must be absent during untraced timing; use export-reference.ps1.");
        if (settings.Profile == "canonical" && Vector.IsHardwareAccelerated)
            throw new InvalidOperationException("Canonical export requires hardware intrinsics disabled.");

        List<Fixture> fixtures = LoadFixtures(settings);
        BundleWriter writer = new(settings.Output);
        string modelsDirectory = Path.Combine(settings.Replay ?? settings.ReferenceRepo, "models");
        if (settings.Replay is not null) ValidateReplayModels(settings.Replay);
        ModelAsset[] assets =
        [
            writer.CopyModel("det", Path.Combine(modelsDirectory, "det.onnx"), "models/det.onnx"),
            writer.CopyModel("cls", Path.Combine(modelsDirectory, "cls.onnx"), "models/cls.onnx"),
            writer.CopyModel("rec", Path.Combine(modelsDirectory, "rec.onnx"), "models/rec.onnx"),
            writer.CopyModel("dictionary", Path.Combine(modelsDirectory, "ppocr_keys.txt"), "models/ppocr_keys.txt")
        ];
        object provenance = Provenance.Capture(settings.ReferenceRepo, settings.ExporterSource, writer.RootPath);
        byte[] dictionary = File.ReadAllBytes(Path.Combine(writer.RootPath, "models", "ppocr_keys.txt"));
        string[] labels = ReadLabels(dictionary);
        PaddleOcrOptions options = Options();
        Stopwatch load = Stopwatch.StartNew();
        using Model det = Model.Load(File.ReadAllBytes(Path.Combine(writer.RootPath, "models", "det.onnx")));
        using Model cls = Model.Load(File.ReadAllBytes(Path.Combine(writer.RootPath, "models", "cls.onnx")));
        using Model rec = Model.Load(File.ReadAllBytes(Path.Combine(writer.RootPath, "models", "rec.onnx")));
        using PaddleOcrAll pipeline = new(det, cls, rec, dictionary, options);
        load.Stop();
        long loadedWorkingSet = WorkingSet();
        List<object> measurements = [];
        List<PaddleOcrResult> publicResults = [];

        // Untraced timing is completed before diagnostic sessions and file writes.
        foreach (Fixture fixture in fixtures)
        {
            Stopwatch clock = Stopwatch.StartNew();
            PaddleOcrResult result = pipeline.Run(fixture.Pixels, fixture.Width, fixture.Height, true, fixture.Stride);
            clock.Stop();
            double firstCallMs = clock.Elapsed.TotalMilliseconds;
            publicResults.Add(result);
            for (int i = 0; i < settings.Warmup; i++)
                pipeline.Run(fixture.Pixels, fixture.Width, fixture.Height, true, fixture.Stride);
            List<double> samples = [];
            long allocatedBefore = GC.GetTotalAllocatedBytes(true);
            for (int i = 0; i < settings.Repeat; i++)
            {
                clock.Restart();
                pipeline.Run(fixture.Pixels, fixture.Width, fixture.Height, true, fixture.Stride);
                clock.Stop();
                samples.Add(clock.Elapsed.TotalMilliseconds);
            }
            measurements.Add(new
            {
                fixture.Id, firstCallMs, warmSamplesMs = samples,
                p50Ms = Percentile(samples, 0.5), p95Ms = Percentile(samples, 0.95),
                imagesPerSecond = 1000 / samples.Average(), workingSetBytes = WorkingSet(),
                allocatedBytes = GC.GetTotalAllocatedBytes(true) - allocatedBefore
            });
            Console.WriteLine($"measured {fixture.Id}: first={firstCallMs:F2}ms warm p50={Percentile(samples, 0.5):F2}ms lines={result.Lines.Length}");
        }
        writer.WriteJson("performance", "performance.json", new
        {
            loadMs = load.Elapsed.TotalMilliseconds, loadedWorkingSetBytes = loadedWorkingSet,
            untracedWorkingSetBytes = WorkingSet(), untracedPeakWorkingSetBytes = PeakWorkingSet(),
            settings.Warmup, settings.Repeat, cases = measurements,
            firstCallSemantics = "First call for each fixture in a shared process; only the first fixture is process-cold. Model loading is measured separately. No tensor tracing or artifact writes in timed calls."
        });

        List<CaseEntry> cases = [];
        using PaddleOcrClassifier classifier = new(cls, options.Classifier);
        using PaddleOcrRecognizer recognizer = new(rec, dictionary, options.Recognizer);
        for (int i = 0; i < fixtures.Count; i++)
        {
            Fixture fixture = fixtures[i];
            List<Artifact> artifacts = [];
            string directory = $"cases/{fixture.Id}";
            artifacts.Add(writer.WriteJson("input.metadata", $"{directory}/input.json", new
            {
                fixture.Width, fixture.Height, fixture.Stride, pixelFormat = "bgr24", fixture.Origin
            }));
            if (fixture.Encoded is not null)
                artifacts.Add(writer.WriteBytes("input.encoded", $"{directory}/source{fixture.Extension}", fixture.Encoded, [fixture.Encoded.Length], "encoded_image"));
            artifacts.Add(writer.WriteBytes("input.pixels", $"{directory}/input.bgr.bin", fixture.Pixels, [fixture.Height, fixture.Stride], "bgr24_rows"));

            PaddleOcrDetectionBox[] boxes = CaptureDetector(writer, artifacts, directory, fixture, det, settings);
            PaddleOcrResult result = publicResults[i];
            if (boxes.Length != result.Lines.Length || boxes.Where((box, line) => box != result.Lines[line].Box).Any())
                throw new InvalidDataException($"Diagnostic detector differs from public pipeline for {fixture.Id}; baseline was not completed.");
            int tracedLines = settings.MaxTracedLines == 0 ? boxes.Length : Math.Min(boxes.Length, settings.MaxTracedLines);
            for (int line = 0; line < boxes.Length; line++)
            {
                string stage = $"line.{line:D3}";
                byte[] crop = PPOCRCrop.Extract(fixture.Pixels, fixture.Width, fixture.Height, fixture.Stride, boxes[line], out int width, out int height);
                artifacts.Add(writer.WriteBytes($"{stage}.crop", $"{directory}/{stage}/crop.bgr.bin", crop, [height, width, 3], "bgr24"));
                if (line >= tracedLines) continue;
                PaddleOcrClassificationResult orientation = classifier.Classify(crop, width, height);
                CaptureClassifier(writer, artifacts, directory, stage, crop, width, height, cls, settings, orientation);
                int rotation = orientation.Label == 1 && orientation.Score > options.ClassifierThreshold ? 180 : 0;
                if (rotation != result.Lines[line].AppliedRotationDegrees)
                    throw new InvalidDataException($"Diagnostic orientation differs for {fixture.Id}/{stage}.");
                if (rotation == 180) PPOCRCrop.Rotate180(crop, width, height);
                artifacts.Add(writer.WriteBytes($"{stage}.rec.pixels", $"{directory}/{stage}/rec.bgr.bin", crop, [height, width, 3], "bgr24"));
                PaddleOcrRecognitionResult dense = CaptureRecognizer(writer, artifacts, directory, stage, crop, width, height, rec, labels, settings);
                PaddleOcrRecognitionResult compact = recognizer.Recognize(crop, width, height, true);
                if (compact.Text != result.Lines[line].Text)
                    throw new InvalidDataException($"Diagnostic recognizer differs for {fixture.Id}/{stage}.");
                VerifyRecognitionStructure(dense, compact, $"{fixture.Id}/{stage}");
                artifacts.Add(writer.WriteJson($"{stage}.rec.public", $"{directory}/{stage}/rec-public.json", RecognitionJson(compact)));
            }
            artifacts.Add(writer.WriteJson("result.public", $"{directory}/result.json", new
            {
                result.DetectedCount, result.DetectorResizedWidth, result.DetectorResizedHeight, result.Text,
                lines = result.Lines.Select(line => new
                {
                    box = BoxJson(line.Box), line.Text, line.RecognitionScore, line.ClassificationScore,
                    line.ClassificationLabel, line.AppliedRotationDegrees, line.EmittedCount,
                    line.RecognitionContentWidth, line.RecognitionTensorWidth, line.RecognitionTimeSteps,
                    ctcSpans = line.CtcSpans
                }).ToArray()
            }));
            cases.Add(new CaseEntry(fixture.Id, "ocr", artifacts));
            Console.WriteLine($"captured {fixture.Id}: {artifacts.Count} artifacts; traced {tracedLines}/{boxes.Length} lines");
        }
        var inventory = new[] { (Id: "det", Value: det), (Id: "cls", Value: cls), (Id: "rec", Value: rec) }
            .Select(model => new
            {
                model.Id, nodes = model.Value.Nodes.Select((node, index) => new
                {
                    index, node.Name, node.OpType, node.Domain, node.Opset, inputs = node.Inputs, outputs = node.Outputs
                }).ToArray()
            }).ToArray();
        writer.WriteJson("graph_inventory", "graph-inventory.json", inventory);
        string assemblyPath = typeof(Model).Assembly.Location;
        writer.WriteManifest(new
        {
            schemaVersion = 1,
            contract = new
            {
                model = "tiny", profile = settings.Profile, backend = "cpu", pixelFormat = "bgr24",
                options, logicalInputLayout = "nchw", tensorEncoding = "ieee754_f32_little_endian",
                decoder = "SixLabors.ImageSharp 3.1.11; replay consumes frozen BGR bytes without decoding",
                nodes = settings.Nodes, settings.MaxTracedLines, returnCtcAlignment = true,
                publicScoreSemantics = "Existing compact MatMul/ArgMax maxima, without probability recalibration; preserved as returned by C#.",
                denseScoreSemantics = "Mean probability at the first frame of each emitted CTC token; final Softmax graph output."
            },
            models = assets, cases, provenance,
            referenceAssembly = new { name = typeof(Model).Assembly.FullName, sha256 = HexHash(File.ReadAllBytes(assemblyPath)) },
            dictionary = new { labels = labels.Length, classCount = labels.Length + 2, blankClass = 0, spaceClass = labels.Length + 1 },
            diagnostics = new { graphInventory = "graph-inventory.json", performance = "performance.json" }
        });
        Console.WriteLine($"saved complete reference bundle: {writer.RootPath}");
    }

    private static PaddleOcrDetectionBox[] CaptureDetector(BundleWriter writer, List<Artifact> artifacts,
        string directory, Fixture fixture, Model model, Settings settings)
    {
        byte[] pixels = fixture.Pixels;
        int width = fixture.Width, height = fixture.Height, stride = fixture.Stride;
        if (width + (long)height < 64)
        {
            int paddedWidth = Math.Max(32, width), paddedHeight = Math.Max(32, height);
            byte[] padded = new byte[checked(paddedWidth * paddedHeight * 3)];
            for (int y = 0; y < height; y++)
                pixels.AsSpan(y * stride, width * 3).CopyTo(padded.AsSpan(y * paddedWidth * 3));
            pixels = padded; width = paddedWidth; height = paddedHeight; stride = width * 3;
        }
        var size = PPOCRPreprocess.ComputeDetSize(width, height, 960);
        float[] input = new float[checked(3 * size.Width * size.Height)];
        PPOCRPreprocess.Det(pixels, width, height, stride, size.Width, size.Height, input);
        artifacts.Add(writer.WriteFloats("det.input", $"{directory}/det/input.f32.bin", input, [1, 3, size.Height, size.Width], "nchw"));
        using InferenceSession session = new(model, [1, 3, size.Height, size.Width]);
        ReadOnlySpan<float> prediction = RunGraph(writer, artifacts, directory, "det", session, input, settings);
        artifacts.Add(writer.WriteFloats("det.probabilities", $"{directory}/det/probabilities.f32.bin", prediction, session.OutputShape.Dimensions, Layout(session)));
        DbPostprocess.Workspace workspace = new();
        PaddleOcrDetectionBox[] boxes = DbPostprocess.Run(prediction, size.Width, size.Height, Options().Detector,
            fixture.Width, fixture.Height, size.WidthRatio, size.HeightRatio, workspace);
        artifacts.Add(writer.WriteJson("det.boxes", $"{directory}/det/boxes.json", new
        {
            resizedWidth = size.Width, resizedHeight = size.Height, size.WidthRatio, size.HeightRatio,
            boxes = boxes.Select(BoxJson).ToArray()
        }));
        return boxes;
    }

    private static void CaptureClassifier(BundleWriter writer, List<Artifact> artifacts, string directory,
        string stage, byte[] crop, int width, int height, Model model, Settings settings, PaddleOcrClassificationResult orientation)
    {
        float[] input = new float[3 * 80 * 160];
        int contentWidth = PPOCRPreprocess.Cls(crop, width, height, width * 3, input);
        string name = $"{stage}.cls";
        artifacts.Add(writer.WriteFloats($"{name}.input", $"{directory}/{name}/input.f32.bin", input, [1, 3, 80, 160], "nchw"));
        using InferenceSession session = new(model, [1, 3, 80, 160]);
        ReadOnlySpan<float> prediction = RunGraph(writer, artifacts, directory, name, session, input, settings);
        artifacts.Add(writer.WriteFloats($"{name}.probabilities", $"{directory}/{name}/probabilities.f32.bin", prediction, session.OutputShape.Dimensions, Layout(session)));
        artifacts.Add(writer.WriteJson($"{name}.public", $"{directory}/{name}/public.json", new
        {
            orientation.Label, orientation.Score, orientation.ResizedWidth, orientation.OrientationDegrees,
            denseContentWidth = contentWidth
        }));
    }

    private static PaddleOcrRecognitionResult CaptureRecognizer(BundleWriter writer, List<Artifact> artifacts, string directory,
        string stage, byte[] crop, int width, int height, Model model, string[] labels, Settings settings)
    {
        long natural = ((long)48 * width + height - 1) / height;
        int tensorWidth = checked((int)Math.Max(32, ((natural + 31) / 32) * 32));
        float[] input = new float[checked(3 * 48 * tensorWidth)];
        int contentWidth = PPOCRPreprocess.Rec(crop, width, height, width * 3, tensorWidth, input);
        string name = $"{stage}.rec";
        artifacts.Add(writer.WriteFloats($"{name}.input", $"{directory}/{name}/input.f32.bin", input, [1, 3, 48, tensorWidth], "nchw"));
        using InferenceSession session = new(model, [1, 3, 48, tensorWidth]);
        ReadOnlySpan<float> prediction = RunGraph(writer, artifacts, directory, name, session, input, settings);
        artifacts.Add(writer.WriteFloats($"{name}.probabilities", $"{directory}/{name}/probabilities.f32.bin", prediction, session.OutputShape.Dimensions, Layout(session)));
        int classes = labels.Length + 2;
        if (prediction.Length % classes != 0 || session.OutputShape.Dimensions[^1] != classes)
            throw new InvalidDataException("Unexpected recognition output shape.");
        PaddleOcrRecognitionResult decoded = PaddleOcrRecognizer.DecodeCtc(labels, labels.Max(label => label.Length), true,
            prediction, [], [], prediction.Length / classes, contentWidth, tensorWidth, false);
        artifacts.Add(writer.WriteJson($"{name}.decoded_dense", $"{directory}/{name}/decoded-dense.json", RecognitionJson(decoded)));
        return decoded;
    }

    private static ReadOnlySpan<float> RunGraph(BundleWriter writer, List<Artifact> artifacts,
        string directory, string stage, InferenceSession session, float[] input, Settings settings)
    {
        Model model = session.Model;
        HashSet<int> requested = [0, model.Nodes.Length / 2, model.Nodes.Length - 1, model.Nodes.Length - 2];
        HashSet<int> checkpoints = [];
        for (int first = 0; first < model.Nodes.Length;)
        {
            int sink = first + session.CompiledModel.FusedSkip(first);
            if (requested.Any(index => index >= first && index <= sink)) checkpoints.Add(sink);
            first = sink + 1;
        }
        return session.RunWithTrace(input, (index, nodeName, opType, shape, nhwc, values) =>
        {
            if (settings.Nodes != "all" && !checkpoints.Contains(index)) return;
            string id = $"{stage}.node.{index:D4}";
            artifacts.Add(writer.WriteFloats(id, $"{directory}/{stage}/node-{index:D4}.f32.bin", values,
                shape.ToArray(), nhwc ? "nhwc" : shape.Length == 4 ? "nchw" : "row_major"));
        });
    }

    private static string Layout(InferenceSession session) =>
        session.CompiledModel.GetTensor(session.CompiledModel.OutputIndex).IsNhwc ? "nhwc" :
        session.OutputShape.Rank == 4 ? "nchw" : "row_major";

    private static object RecognitionJson(PaddleOcrRecognitionResult result) => new
    {
        result.Text, result.Score, result.EmittedCount, result.ResizedWidth, result.TensorWidth,
        result.TimeSteps, result.CtcSpans
    };

    private static void VerifyRecognitionStructure(PaddleOcrRecognitionResult dense, PaddleOcrRecognitionResult compact, string location)
    {
        bool same = dense.Text == compact.Text && dense.EmittedCount == compact.EmittedCount &&
            dense.ResizedWidth == compact.ResizedWidth && dense.TensorWidth == compact.TensorWidth && dense.TimeSteps == compact.TimeSteps;
        var left = dense.CtcSpans ?? [];
        var right = compact.CtcSpans ?? [];
        same &= left.Length == right.Length && left.Select(span => (span.Text, span.StartColumn, span.EndColumn))
            .SequenceEqual(right.Select(span => (span.Text, span.StartColumn, span.EndColumn)));
        if (!same) throw new InvalidDataException($"Dense and public recognition structure differ at {location}; scores have separate contracts, but text/alignment must agree.");
    }

    private static object BoxJson(PaddleOcrDetectionBox box) => new
    {
        points = new[] { new[] { box.X1, box.Y1 }, new[] { box.X2, box.Y2 }, new[] { box.X3, box.Y3 }, new[] { box.X4, box.Y4 } }, box.Score
    };

    private static PaddleOcrOptions Options() => new()
    {
        LineWorkerCount = 1, DetIntraOpThreads = 1, PreprocessWorkerCount = 1,
        GpuCtcIntraOpThreads = 1, RecBatchLines = 1, UseDirectionClassification = true, ClassifierThreshold = 0,
        Detector = new() { Backend = OcrBackend.Cpu, BoxThreshold = 0.4f, MaxPooledSessions = 1 },
        Classifier = new() { Backend = OcrBackend.Cpu, MaxPooledSessions = 1 },
        Recognizer = new() { Backend = OcrBackend.Cpu, AdaptiveWidth = true, TargetWidth = 320, MaxPooledSessions = 1 }
    };

    private static string[] ReadLabels(byte[] dictionary)
    {
        string text = new UTF8Encoding(false, true).GetString(dictionary);
        if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
        string[] raw = text.Split('\n');
        if (raw.Length > 0 && raw[^1].Length == 0) raw = raw[..^1];
        return raw.Select(label => label.EndsWith('\r') ? label[..^1] : label).ToArray();
    }

    private static List<Fixture> LoadFixtures(Settings settings)
    {
        if (settings.Replay is not null) return ReplayFixtures(settings.Replay);
        if (settings.Input is null)
        {
            Fixture sample = Decode(Path.Combine(settings.ReferenceRepo, "examples", "sample.jpg"), "sample");
            byte[] rotated = sample.Pixels.ToArray();
            // A 180-degree rotation reverses packed BGR pixels, preserving each channel triplet.
            for (int i = 0, j = rotated.Length - 3; i < j; i += 3, j -= 3)
                for (int channel = 0; channel < 3; channel++)
                    (rotated[i + channel], rotated[j + channel]) = (rotated[j + channel], rotated[i + channel]);
            return
            [
                sample,
                new("sample-180", rotated, sample.Width, sample.Height, sample.Stride, "sample.jpg rotated 180 degrees from frozen decoded bytes", null, ""),
                new("blank", Enumerable.Repeat((byte)255, 32 * 32 * 3).ToArray(), 32, 32, 96, "constant white 32x32", null, ""),
                new("tiny", [31, 127, 223], 1, 1, 3, "constant BGR pixel; detector padding case", null, "")
            ];
        }
        string[] files = File.Exists(settings.Input) ? [settings.Input] : Directory.GetFiles(settings.Input)
            .Where(path => new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" }.Contains(Path.GetExtension(path).ToLowerInvariant()))
            .OrderBy(path => Path.GetFileName(path), StringComparer.Ordinal).Take(settings.Count).ToArray();
        if (files.Length == 0) throw new ArgumentException("No supported input images found.");
        return files.Select((path, index) => Decode(path, $"image-{index:D3}")).ToList();
    }

    private static List<Fixture> ReplayFixtures(string root)
    {
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "manifest.json")));
        if (manifest.RootElement.GetProperty("schema_version").GetInt32() != 1)
            throw new InvalidDataException("Unsupported replay bundle schema.");
        List<Fixture> fixtures = [];
        HashSet<string> ids = [];
        foreach (JsonElement item in manifest.RootElement.GetProperty("cases").EnumerateArray())
        {
            string id = item.GetProperty("id").GetString()!;
            BundleWriter.NormalizeRelativePath(id);
            if (id.Contains('/') || !ids.Add(id) || item.GetProperty("kind").GetString() != "ocr")
                throw new InvalidDataException("Replay requires unique OCR case IDs.");
            JsonElement[] artifacts = item.GetProperty("artifacts").EnumerateArray().ToArray();
            JsonElement metadata = artifacts.Single(artifact => artifact.GetProperty("id").GetString() == "input.metadata");
            using JsonDocument input = JsonDocument.Parse(ReadBundleFile(root, metadata));
            JsonElement fields = input.RootElement;
            if (fields.GetProperty("pixel_format").GetString() != "bgr24")
                throw new InvalidDataException("Replay requires BGR24 input.");
            int width = fields.GetProperty("width").GetInt32(), height = fields.GetProperty("height").GetInt32(), stride = fields.GetProperty("stride").GetInt32();
            byte[] pixels = ReadBundleFile(root, artifacts.Single(artifact => artifact.GetProperty("id").GetString() == "input.pixels"));
            if (width <= 0 || height <= 0 || stride < checked(width * 3) || pixels.Length != checked(height * stride))
                throw new InvalidDataException($"Invalid replay pixel buffer for {id}.");
            JsonElement encoded = artifacts.FirstOrDefault(artifact => artifact.GetProperty("id").GetString() == "input.encoded");
            fixtures.Add(new(id, pixels, width, height, stride, fields.GetProperty("origin").GetString()!,
                encoded.ValueKind == JsonValueKind.Undefined ? null : ReadBundleFile(root, encoded),
                encoded.ValueKind == JsonValueKind.Undefined ? "" : Path.GetExtension(encoded.GetProperty("path").GetString()!)));
        }
        if (fixtures.Count == 0) throw new InvalidDataException("Replay bundle contains no cases.");
        return fixtures;
    }

    private static void ValidateReplayModels(string root)
    {
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "manifest.json")));
        JsonElement[] models = manifest.RootElement.GetProperty("models").EnumerateArray().ToArray();
        foreach ((string id, string path) in new[] { ("det", "models/det.onnx"), ("cls", "models/cls.onnx"), ("rec", "models/rec.onnx"), ("dictionary", "models/ppocr_keys.txt") })
        {
            JsonElement asset = models.Single(model => model.GetProperty("id").GetString() == id);
            if (asset.GetProperty("path").GetString() != path) throw new InvalidDataException($"Unexpected replay model path for {id}.");
            _ = ReadBundleFile(root, asset);
        }
    }

    private static byte[] ReadBundleFile(string root, JsonElement descriptor)
    {
        string path = BundleWriter.DestinationPath(root, descriptor.GetProperty("path").GetString()!);
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.LongLength != descriptor.GetProperty("bytes").GetInt64() || HexHash(bytes) != descriptor.GetProperty("sha256").GetString())
            throw new InvalidDataException($"Corrupted replay input/model: {descriptor.GetProperty("path").GetString()}");
        return bytes;
    }

    private static Fixture Decode(string path, string id)
    {
        byte[] encoded = File.ReadAllBytes(path);
        using Image<Bgr24> image = Image.Load<Bgr24>(encoded);
        byte[] pixels = new byte[checked(image.Width * image.Height * 3)];
        image.CopyPixelDataTo(pixels);
        return new(id, pixels, image.Width, image.Height, image.Width * 3, Path.GetFileName(path), encoded, Path.GetExtension(path).ToLowerInvariant());
    }

    private static void RequireEnvironment(string name, string expected)
    {
        if (Environment.GetEnvironmentVariable(name) != expected)
            throw new InvalidOperationException($"{name} must be {expected}; launch through export-reference.ps1.");
    }
    private static string HexHash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static long WorkingSet() { using Process process = Process.GetCurrentProcess(); return process.WorkingSet64; }
    private static long PeakWorkingSet() { using Process process = Process.GetCurrentProcess(); return process.PeakWorkingSet64; }
    private static double Percentile(List<double> values, double fraction)
    {
        double[] sorted = values.Order().ToArray();
        double position = (sorted.Length - 1) * fraction;
        int lower = (int)Math.Floor(position), upper = (int)Math.Ceiling(position);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private sealed record Fixture(string Id, byte[] Pixels, int Width, int Height, int Stride, string Origin, byte[]? Encoded, string Extension);
    private sealed record Settings(string ReferenceRepo, string Output, string ExporterSource, string? Input, string? Replay,
        string Profile, string Nodes, int MaxTracedLines, int Count, int Warmup, int Repeat)
    {
        public static Settings Parse(string[] args)
        {
            Dictionary<string, string> values = [];
            HashSet<string> known = ["--reference-repo", "--out", "--exporter-source", "--input", "--replay", "--profile", "--nodes", "--max-traced-lines", "--count", "--warmup", "--repeat"];
            for (int i = 0; i < args.Length; i++)
            {
                string name = args[i];
                if (!known.Contains(name) || i + 1 == args.Length || !values.TryAdd(name, args[++i]))
                    throw new ArgumentException($"Invalid, duplicate, or missing argument: {name}");
            }
            string Required(string name) => values.GetValueOrDefault(name) ?? throw new ArgumentException($"Required: {name}");
            int Number(string name, int fallback, int minimum) =>
                int.TryParse(values.GetValueOrDefault(name, fallback.ToString()), out int value) && value >= minimum
                    ? value : throw new ArgumentException($"{name} must be an integer >= {minimum}");
            string profile = values.GetValueOrDefault("--profile", "canonical"), nodes = values.GetValueOrDefault("--nodes", "checkpoints");
            if (profile is not ("canonical" or "cpu") || nodes is not ("checkpoints" or "all"))
                throw new ArgumentException("Invalid --profile or --nodes.");
            if (values.ContainsKey("--input") && values.ContainsKey("--replay")) throw new ArgumentException("Choose --input or --replay.");
            return new(Path.GetFullPath(Required("--reference-repo")), Path.GetFullPath(Required("--out")),
                Path.GetFullPath(Required("--exporter-source")), values.TryGetValue("--input", out string? input) ? Path.GetFullPath(input) : null,
                values.TryGetValue("--replay", out string? replay) ? Path.GetFullPath(replay) : null,
                profile, nodes, Number("--max-traced-lines", 3, 0), Number("--count", 20, 1), Number("--warmup", 1, 0), Number("--repeat", 3, 1));
        }
    }
}
