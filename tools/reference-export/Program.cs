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
            if (args.Length > 0 && args[0] == "--export-imports") return ImportTools.Export(args[1..]);
            if (args.Length > 0 && args[0] == "--benchmark-imports") return ImportTools.Benchmark(args[1..]);
            if (args.Length > 0 && args[0] == "--benchmark-cls") return ClsTools.Benchmark(args[1..]);
            if (args.Length > 0 && args[0] == "--benchmark-step12") return Step12Tools.Benchmark(args[1..]);
            if (args.Length > 0 && args[0] == "--benchmark-ocr") return OcrBenchmark.Run(args[1..]);
            if (args.Length > 0 && args[0] == "--validate-ocr-public") return PublicBenchmarkValidation.Run(args[1..]);
            if (args.Length == 1 && args[0] == "--verify-public-contracts") return PublicSemanticContractTests.Run();
            if (args.Length > 0 && args[0] == "--export-geometry") return GeometryFixtures.Export(args[1..]);
            if (args.Length > 0 && args[0] == "--export-rec-fixtures") return RecFixtures.Export(args[1..]);
            if (args.Length > 0 && args[0] == "--export-rec-ops") return RecFixtures.ExportOperators(args[1..]);
            if (args.Length > 0 && args[0] == "--export-scalar-rec-ops") return RecFixtures.ExportScalarOperators(args[1..]);
            if (args.Length == 1 && args[0] == "--verify-rec-contracts")
                return RecContractTests.Run((dense, compact, contentWidth, tensorWidth, timeSteps, location) =>
                    VerifyRecognitionStructure(dense, compact, new(contentWidth, tensorWidth, timeSteps), location));
            if (args.Length == 1 && args[0] == "--verify-export-contracts") return ExportContractTests.Run();
            if (args.Length == 0 || args.Contains("--help"))
            {
                Console.WriteLine("Use tools/export-reference.ps1 -Out <new-directory> [-InputPath <image-or-directory> | -InputBundle <verified-frozen-pixels> | -ReplayBundle <same-model-frozen-contract>] [-CaptureScope ocr|step12] [-Model tiny|small|medium] [-DetectorModel tiny|small|medium for step12] [-FixtureSet default|compact|expanded] [-Profile canonical|cpu] [-Nodes checkpoints|all] [-MaxTracedLines 3] [-Count 20] [-Warmup 1] [-Repeat 3].");
                return args.Length == 0 ? 2 : 0;
            }
            Settings settings = Settings.Parse(args);
            if (settings.CaptureScope == "step12") Step12Tools.Export(settings);
            else Export(settings);
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
        ModelPreset preset = ModelPreset.For(settings.Model);
        Dictionary<string, string> sources = settings.Replay is not null
            ? ReplayModels(settings.Replay, settings.Model)
            : new() { ["det"] = Path.Combine(settings.ReferenceRepo, "models", preset.DetectorFile),
                ["cls"] = Path.Combine(settings.ReferenceRepo, "models", "cls.onnx"),
                ["rec"] = Path.Combine(settings.ReferenceRepo, "models", preset.RecognizerFile),
                ["dictionary"] = Path.Combine(settings.ReferenceRepo, "models", preset.DictionaryFile) };
        ModelAsset[] assets = [writer.CopyModel("det", sources["det"], "models/det.onnx"),
            writer.CopyModel("cls", sources["cls"], "models/cls.onnx"),
            writer.CopyModel("rec", sources["rec"], "models/rec.onnx"),
            writer.CopyModel("dictionary", sources["dictionary"], "models/" + preset.DictionaryFile)];
        object provenance = Provenance.Capture(settings.ReferenceRepo, settings.ExporterSource, writer.RootPath);
        byte[] dictionary = File.ReadAllBytes(Path.Combine(writer.RootPath, "models", preset.DictionaryFile));
        string[] labels = ReadLabels(dictionary);
        PaddleOcrOptions options = settings.EffectiveOptions();
        Stopwatch load = Stopwatch.StartNew();
        Dictionary<string, ModelInspectionSnapshot> inspected = [];
        Model Load(string id)
        {
            using FileStream stream = File.OpenRead(Path.Combine(writer.RootPath, "models", $"{id}.onnx"));
            return Model.Load(stream, snapshot => { if (snapshot.Stage == "normalized") inspected.Add(id, snapshot); });
        }
        using Model det = Load("det");
        using Model cls = Load("cls");
        using Model rec = Load("rec");
        using (InferenceSession shapeProbe = new(rec, [1, 3, 48, options.Recognizer.AdaptiveWidth ? 32 : options.Recognizer.TargetWidth], 1))
        {
            int[] dimensions = shapeProbe.OutputShape.Dimensions.ToArray();
            if (dimensions.Length != 3 || dimensions[0] != 1 || dimensions[2] != labels.Length + 2)
                throw new InvalidDataException("Selected REC model vocabulary does not match the dictionary.");
        }
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
        Artifact performanceFile = writer.WriteJson("performance", "performance.json", new
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

            PaddleOcrDetectionBox[] boxes = CaptureDetector(writer, artifacts, directory, fixture, det, settings, options.Detector);
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
                PaddleOcrRecognitionResult dense = CaptureRecognizer(writer, artifacts, directory, stage, crop, width, height, rec, labels, settings, out RecognitionDimensions dimensions);
                PaddleOcrRecognitionResult compact = recognizer.Recognize(crop, width, height, true);
                if (compact.Text != result.Lines[line].Text)
                    throw new InvalidDataException($"Diagnostic recognizer differs for {fixture.Id}/{stage}.");
                VerifyRecognitionStructure(dense, compact, dimensions, $"{fixture.Id}/{stage}");
                artifacts.Add(writer.WriteJson($"{stage}.rec.public", $"{directory}/{stage}/rec-public.json", RecognitionJson(compact)));
            }
            artifacts.Add(writer.WriteJson("result.public", $"{directory}/result.json", PublicOcrJson(result)));
            cases.Add(new CaseEntry(fixture.Id, "ocr", artifacts));
            Console.WriteLine($"captured {fixture.Id}: {artifacts.Count} artifacts; traced {tracedLines}/{boxes.Length} lines");
        }
        var inventory = new[] { "det", "cls", "rec" }
            .Select(id => new
            {
                id, inputs = inspected[id].Inputs, outputs = inspected[id].Outputs,
                nodes = inspected[id].Nodes.Select((node, index) => new
                {
                    index, node.Name, node.OpType, node.Domain, node.Opset, inputs = node.Inputs, outputs = node.Outputs
                }).ToArray()
            }).ToArray();
        Artifact inventoryFile = writer.WriteJson("graph_inventory", "graph-inventory.json", inventory);
        string assemblyPath = typeof(Model).Assembly.Location;
        writer.WriteManifest(new
        {
            schemaVersion = 1,
            contract = new
            {
                model = settings.Model, captureScope = "ocr", inputOrigin = "frozen_bgr_pixels",
                inventorySemantics = "lineage_keys_v1", profile = settings.Profile, backend = "cpu", pixelFormat = "bgr24",
                options, logicalInputLayout = "nchw", tensorEncoding = "ieee754_f32_little_endian",
                decoder = "SixLabors.ImageSharp 3.1.11; replay consumes frozen BGR bytes without decoding",
                nodes = settings.Nodes, settings.MaxTracedLines, returnCtcAlignment = true,
                publicScoreSemantics = "Existing compact MatMul/ArgMax maxima, without probability recalibration; preserved as returned by C#.",
                denseScoreSemantics = "Mean probability at the first frame of each emitted CTC token; final Softmax graph output."
            },
            models = assets, cases, provenance,
            auxiliaryFiles = ImportTools.ProvenanceFiles((ReferenceProvenance)provenance)
                .Concat(new[] { ImportTools.Auxiliary(performanceFile), ImportTools.Auxiliary(inventoryFile) }).ToArray(),
            referenceAssembly = new { name = typeof(Model).Assembly.FullName, sha256 = HexHash(File.ReadAllBytes(assemblyPath)) },
            dictionary = new { labels = labels.Length, classCount = labels.Length + 2, blankClass = 0, spaceClass = labels.Length + 1 },
            diagnostics = new { graphInventory = "graph-inventory.json", performance = "performance.json" }
        });
        Console.WriteLine($"saved complete reference bundle: {writer.RootPath}");
    }

    internal static PaddleOcrDetectionBox[] CaptureDetector(BundleWriter writer, List<Artifact> artifacts,
        string directory, Fixture fixture, Model model, Settings settings, PaddleOcrDetectorOptions? detectorOptions = null)
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
        var options = detectorOptions ?? settings.EffectiveOptions().Detector;
        var size = PPOCRPreprocess.ComputeDetSize(width, height, options.LimitSideLength);
        float[] input = new float[checked(3 * size.Width * size.Height)];
        PPOCRPreprocess.Det(pixels, width, height, stride, size.Width, size.Height, input);
        artifacts.Add(writer.WriteFloats("det.input", $"{directory}/det/input.f32.bin", input, [1, 3, size.Height, size.Width], "nchw"));
        using InferenceSession session = new(model, [1, 3, size.Height, size.Width]);
        ReadOnlySpan<float> prediction = RunGraph(writer, artifacts, directory, "det", session, input, settings);
        artifacts.Add(writer.WriteFloats("det.probabilities", $"{directory}/det/probabilities.f32.bin", prediction, session.OutputShape.Dimensions, Layout(session)));
        DbPostprocess.Workspace workspace = new();
        PaddleOcrDetectionBox[] boxes = DbPostprocess.Run(prediction, size.Width, size.Height, options,
            fixture.Width, fixture.Height, size.WidthRatio, size.HeightRatio, workspace);
        artifacts.Add(writer.WriteJson("det.boxes", $"{directory}/det/boxes.json", new
        {
            resizedWidth = size.Width, resizedHeight = size.Height, size.WidthRatio, size.HeightRatio,
            boxes = boxes.Select(BoxJson).ToArray()
        }));
        return boxes;
    }

    internal static void CaptureClassifier(BundleWriter writer, List<Artifact> artifacts, string directory,
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
        string stage, byte[] crop, int width, int height, Model model, string[] labels, Settings settings, out RecognitionDimensions dimensions)
    {
        long natural = ((long)48 * width + height - 1) / height;
        PaddleOcrRecognizerOptions options = settings.EffectiveOptions().Recognizer;
        int tensorWidth = options.AdaptiveWidth ? checked((int)Math.Max(32, ((natural + 31) / 32) * 32)) : options.TargetWidth;
        float[] input = new float[checked(3 * 48 * tensorWidth)];
        int contentWidth = PPOCRPreprocess.Rec(crop, width, height, width * 3, tensorWidth, input);
        if (contentWidth != Math.Min(natural, tensorWidth))
            throw new InvalidDataException("Recognition content width differs from crop-derived metadata.");
        string name = $"{stage}.rec";
        artifacts.Add(writer.WriteFloats($"{name}.input", $"{directory}/{name}/input.f32.bin", input, [1, 3, 48, tensorWidth], "nchw"));
        using InferenceSession session = new(model, [1, 3, 48, tensorWidth]);
        ReadOnlySpan<float> prediction = RunGraph(writer, artifacts, directory, name, session, input, settings);
        artifacts.Add(writer.WriteFloats($"{name}.probabilities", $"{directory}/{name}/probabilities.f32.bin", prediction, session.OutputShape.Dimensions, Layout(session)));
        int classes = labels.Length + 2;
        int[] outputShape = session.OutputShape.Dimensions.ToArray();
        if (outputShape.Length != 3 || outputShape[0] != 1 || outputShape[1] <= 0 || outputShape[2] != classes || prediction.Length != checked(outputShape[1] * classes))
            throw new InvalidDataException("Unexpected recognition output shape.");
        dimensions = new(contentWidth, tensorWidth, outputShape[1]);
        PaddleOcrRecognitionResult decoded = PaddleOcrRecognizer.DecodeCtc(labels, labels.Max(label => label.Length), true,
            prediction, [], [], dimensions.TimeSteps, contentWidth, tensorWidth, false);
        ValidateRecognitionForm(decoded, dimensions, $"{directory}/{name}/dense");
        artifacts.Add(writer.WriteJson($"{name}.decoded_dense", $"{directory}/{name}/decoded-dense.json", RecognitionJson(decoded)));
        return decoded;
    }

    internal static ReadOnlySpan<float> RunGraph(BundleWriter writer, List<Artifact> artifacts,
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

    internal static object RecognitionJson(PaddleOcrRecognitionResult result) => new
    {
        result.Text, result.Score, result.EmittedCount, result.ResizedWidth, result.TensorWidth,
        result.TimeSteps, result.CtcSpans
    };

    internal static object PublicOcrJson(PaddleOcrResult result) => new
    {
        result.DetectedCount, result.DetectorResizedWidth, result.DetectorResizedHeight, result.Text,
        lines = result.Lines.Select(line => new
        {
            box = BoxJson(line.Box), line.Text, line.RecognitionScore, line.ClassificationScore,
            line.ClassificationLabel, line.AppliedRotationDegrees, line.EmittedCount,
            line.RecognitionContentWidth, line.RecognitionTensorWidth, line.RecognitionTimeSteps,
            ctcSpans = line.CtcSpans
        }).ToArray()
    };

    private readonly record struct RecognitionDimensions(int ContentWidth, int TensorWidth, int TimeSteps);

    private static void VerifyRecognitionStructure(PaddleOcrRecognitionResult dense, PaddleOcrRecognitionResult compact, RecognitionDimensions dimensions, string location)
    {
        // Bias-first compact projection and separately rounded dense MatMul/Add
        // can choose different classes. Validate each native form independently.
        ValidateRecognitionForm(dense, dimensions, location + "/dense");
        ValidateRecognitionForm(compact, dimensions, location + "/compact");
    }

    private static void ValidateRecognitionForm(PaddleOcrRecognitionResult result, RecognitionDimensions dimensions, string location)
    {
        if (dimensions.ContentWidth <= 0 || dimensions.ContentWidth > dimensions.TensorWidth || dimensions.TensorWidth <= 0 || dimensions.TimeSteps <= 0 ||
            result.ResizedWidth != dimensions.ContentWidth || result.TensorWidth != dimensions.TensorWidth || result.TimeSteps != dimensions.TimeSteps ||
            result.Text is null || !float.IsFinite(result.Score) || result.EmittedCount < 0 || result.EmittedCount > dimensions.TimeSteps)
            throw new InvalidDataException($"Recognition dimensions, count or score are invalid at {location}.");
        PaddleOcrCtcSpan[] spans = result.CtcSpans ?? throw new InvalidDataException($"Captured recognition spans are missing at {location}.");
        if (spans.Length != result.EmittedCount || (result.EmittedCount == 0 && (result.Text.Length != 0 || result.Score != 0)))
            throw new InvalidDataException($"Recognition emission/span structure is invalid at {location}.");
        int previousEnd = 0;
        StringBuilder combined = new();
        foreach (PaddleOcrCtcSpan span in spans)
        {
            // Empty dictionary entries are legitimate emitted tokens.
            if (span.Text is null || !float.IsFinite(span.Score) || span.StartColumn < previousEnd ||
                span.StartColumn >= span.EndColumn || span.EndColumn > dimensions.TimeSteps)
                throw new InvalidDataException($"Recognition span bounds or score are invalid at {location}.");
            combined.Append(span.Text);
            previousEnd = span.EndColumn;
        }
        if (!string.Equals(combined.ToString(), result.Text, StringComparison.Ordinal))
            throw new InvalidDataException($"Recognition spans do not reconstruct text at {location}.");
    }

    internal static object BoxJson(PaddleOcrDetectionBox box) => new
    {
        points = new[] { new[] { box.X1, box.Y1 }, new[] { box.X2, box.Y2 }, new[] { box.X3, box.Y3 }, new[] { box.X4, box.Y4 } }, box.Score
    };

    internal static PaddleOcrOptions Options(string model = "tiny") => new()
    {
        LineWorkerCount = 1, DetIntraOpThreads = 1, PreprocessWorkerCount = 1,
        GpuCtcIntraOpThreads = 1, RecBatchLines = 1, UseDirectionClassification = true, ClassifierThreshold = 0,
        Detector = new() { Backend = OcrBackend.Cpu, BoxThreshold = ModelPreset.For(model).BoxThreshold, MaxPooledSessions = 1 },
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

    internal static List<Fixture> LoadFixtures(Settings settings)
    {
        if (settings.Replay is not null) return ReplayFixtures(settings.Replay);
        if (settings.InputBundle is not null) return ReplayFixtures(settings.InputBundle, true);
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

    private static List<Fixture> ReplayFixtures(string root, bool requireOcr = false)
    {
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "manifest.json")));
        if (manifest.RootElement.GetProperty("schema_version").GetInt32() != 1)
            throw new InvalidDataException("Unsupported replay bundle schema.");
        ValidateBundleFiles(root, manifest.RootElement);
        JsonElement contract = manifest.RootElement.GetProperty("contract");
        string caseKind = contract.TryGetProperty("capture_scope", out JsonElement scope) && scope.GetString() == "step12" ? "step12" : "ocr";
        if (requireOcr && caseKind != "ocr") throw new InvalidDataException("InputBundle requires full OCR frozen source pixels.");
        _ = ModelPreset.For(contract.GetProperty("model").GetString()!);
        List<Fixture> fixtures = [];
        HashSet<string> ids = [];
        foreach (JsonElement item in manifest.RootElement.GetProperty("cases").EnumerateArray())
        {
            string id = item.GetProperty("id").GetString()!;
            BundleWriter.NormalizeRelativePath(id);
            if (id.Contains('/') || !ids.Add(id) || item.GetProperty("kind").GetString() != caseKind)
                throw new InvalidDataException("Replay requires unique case IDs matching the capture scope.");
            JsonElement[] artifacts = item.GetProperty("artifacts").EnumerateArray().ToArray();
            JsonElement metadata = artifacts.Single(artifact => artifact.GetProperty("id").GetString() == "input.metadata");
            if (metadata.GetProperty("dtype").GetString() != "json" || metadata.GetProperty("layout").GetString() != "json" || metadata.GetProperty("shape").GetArrayLength() != 0)
                throw new InvalidDataException("Replay source metadata must be a scalar JSON artifact.");
            using JsonDocument input = JsonDocument.Parse(ReadBundleFile(root, metadata));
            JsonElement fields = input.RootElement;
            if (fields.GetProperty("pixel_format").GetString() != "bgr24")
                throw new InvalidDataException("Replay requires BGR24 input.");
            int width = fields.GetProperty("width").GetInt32(), height = fields.GetProperty("height").GetInt32(), stride = fields.GetProperty("stride").GetInt32();
            JsonElement pixelArtifact = artifacts.Single(artifact => artifact.GetProperty("id").GetString() == "input.pixels");
            if (pixelArtifact.GetProperty("dtype").GetString() != "u8" || pixelArtifact.GetProperty("layout").GetString() != "bgr24_rows" ||
                !pixelArtifact.GetProperty("shape").EnumerateArray().Select(value => value.GetInt32()).SequenceEqual(new[] { height, stride }))
                throw new InvalidDataException("Replay source pixel dtype, shape or layout differs from metadata.");
            byte[] pixels = ReadBundleFile(root, pixelArtifact);
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

    private static Dictionary<string, string> ReplayModels(string root, string model)
    {
        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "manifest.json")));
        JsonElement[] models = manifest.RootElement.GetProperty("models").EnumerateArray().ToArray();
        if (manifest.RootElement.GetProperty("contract").GetProperty("model").GetString() != model)
            throw new InvalidDataException("Replay model conflicts with the frozen contract; use --input-bundle for frozen input reuse.");
        if (models.Length != 4) throw new InvalidDataException("OCR replay requires exactly DET, CLS, REC and dictionary assets.");
        Dictionary<string, string> paths = [];
        foreach (string id in new[] { "det", "cls", "rec", "dictionary" })
        {
            JsonElement asset = models.Single(model => model.GetProperty("id").GetString() == id);
            _ = ReadBundleFile(root, asset);
            paths.Add(id, BundleWriter.DestinationPath(root, asset.GetProperty("path").GetString()!));
        }
        return paths;
    }

    internal static byte[] ReadBundleFile(string root, JsonElement descriptor)
    {
        string path = BundleWriter.DestinationPath(root, descriptor.GetProperty("path").GetString()!);
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.LongLength != descriptor.GetProperty("bytes").GetInt64() || HexHash(bytes) != descriptor.GetProperty("sha256").GetString())
            throw new InvalidDataException($"Corrupted replay input/model: {descriptor.GetProperty("path").GetString()}");
        return bytes;
    }

    internal static void ValidateBundleFiles(string root, JsonElement manifest)
    {
        if (manifest.GetProperty("schema_version").GetInt32() != 1) throw new InvalidDataException("Unsupported bundle schema.");
        foreach (JsonElement asset in manifest.GetProperty("models").EnumerateArray()) _ = ReadBundleFile(root, asset);
        if (manifest.TryGetProperty("auxiliary_files", out JsonElement files))
            foreach (JsonElement file in files.EnumerateArray()) _ = ReadBundleFile(root, file);
        foreach (JsonElement item in manifest.GetProperty("cases").EnumerateArray())
            foreach (JsonElement artifact in item.GetProperty("artifacts").EnumerateArray()) _ = ReadBundleFile(root, artifact);
    }

    internal static PaddleOcrOptions ReadOptions(JsonElement contract)
    {
        JsonElement configured = contract.GetProperty("options");
        foreach (string field in new[] { "use_direction_classification", "classifier_threshold", "line_worker_count", "det_intra_op_threads",
            "preprocess_worker_count", "gpu_ctc_intra_op_threads", "rec_batch_lines", "max_crop_pixels", "detector", "classifier", "recognizer" })
            _ = configured.GetProperty(field);
        foreach (string field in new[] { "backend", "limit_side_length", "max_candidates", "bitmap_threshold", "box_threshold", "unclip_ratio", "use_dilation", "max_image_pixels", "max_pooled_sessions" })
            _ = configured.GetProperty("detector").GetProperty(field);
        foreach (string field in new[] { "backend", "max_image_pixels", "max_pooled_sessions" }) _ = configured.GetProperty("classifier").GetProperty(field);
        foreach (string field in new[] { "backend", "adaptive_width", "target_width", "max_image_pixels", "max_pooled_sessions" }) _ = configured.GetProperty("recognizer").GetProperty(field);
        PaddleOcrOptions options = configured.Deserialize<PaddleOcrOptions>(BundleWriter.JsonOptions) ?? throw new InvalidDataException("Missing OCR options.");
        if (options.LineWorkerCount != 1 || options.DetIntraOpThreads != 1 || options.PreprocessWorkerCount != 1 ||
            options.GpuCtcIntraOpThreads != 1 || options.RecBatchLines != 1 || !options.UseDirectionClassification ||
            options.Detector.Backend != OcrBackend.Cpu || options.Classifier.Backend != OcrBackend.Cpu || options.Recognizer.Backend != OcrBackend.Cpu ||
            options.Detector.MaxPooledSessions != 1 || options.Classifier.MaxPooledSessions != 1 || options.Recognizer.MaxPooledSessions != 1 ||
            options.MaxCropPixels <= 0 || options.Detector.MaxImagePixels <= 0 || options.Classifier.MaxImagePixels <= 0 || options.Recognizer.MaxImagePixels <= 0 ||
            options.Detector.LimitSideLength <= 0 || options.Detector.MaxCandidates <= 0 || options.Recognizer.TargetWidth <= 0 ||
            !float.IsFinite(options.ClassifierThreshold) || options.ClassifierThreshold < 0 || options.ClassifierThreshold > 1 ||
            !float.IsFinite(options.Detector.BitmapThreshold) || options.Detector.BitmapThreshold < 0 || options.Detector.BitmapThreshold > 1 ||
            !float.IsFinite(options.Detector.BoxThreshold) || options.Detector.BoxThreshold < 0 || options.Detector.BoxThreshold > 1 ||
            !float.IsFinite(options.Detector.UnclipRatio) || options.Detector.UnclipRatio <= 0)
            throw new InvalidDataException("Frozen OCR options must describe valid single-worker CPU execution.");
        return options;
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

    internal sealed record Fixture(string Id, byte[] Pixels, int Width, int Height, int Stride, string Origin, byte[]? Encoded, string Extension);
    internal sealed record ModelPreset(string Family, string DetectorFile, string RecognizerFile, string DictionaryFile, float BoxThreshold)
    {
        public static ModelPreset For(string family) => family switch
        {
            "tiny" => new(family, "det.onnx", "rec.onnx", "ppocr_keys.txt", .4f),
            "small" or "medium" => new(family, family + "_det.onnx", family + "_rec.onnx", "rec_keys.txt", .45f),
            _ => throw new ArgumentException("Model must be tiny, small or medium.")
        };
    }
    internal sealed record Settings(string ReferenceRepo, string Output, string ExporterSource, string? Input, string? Replay,
        string Profile, string Nodes, int MaxTracedLines, int Count, int Warmup, int Repeat,
        string CaptureScope, string DetectorModel, string FixtureSet, string Model, string? InputBundle, JsonElement? ReplayContract)
    {
        public PaddleOcrOptions EffectiveOptions() => ReplayContract is JsonElement contract && CaptureScope == "ocr" ? ReadOptions(contract) : Options(Model);
        public static Settings Parse(string[] args)
        {
            Dictionary<string, string> values = [];
            HashSet<string> known = ["--reference-repo", "--out", "--exporter-source", "--input", "--input-bundle", "--replay", "--profile", "--nodes", "--max-traced-lines", "--count", "--warmup", "--repeat", "--capture-scope", "--model", "--detector-model", "--fixture-set"];
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
            JsonElement? replayContract = null;
            if (values.TryGetValue("--replay", out string? replayRoot))
            {
                using JsonDocument replayDocument = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(replayRoot, "manifest.json")));
                replayContract = replayDocument.RootElement.GetProperty("contract").Clone();
                JsonElement frozen = replayContract.Value;
                foreach ((string arg, string field) in new[] { ("--capture-scope", "capture_scope"), ("--model", "model") })
                    if (frozen.TryGetProperty(field, out JsonElement value))
                    {
                        if (values.TryGetValue(arg, out string? supplied) && supplied != value.GetString()) throw new ArgumentException($"{arg} conflicts with the replay contract; use --input-bundle for input-only reuse.");
                        values[arg] = value.GetString()!;
                    }
                if (frozen.TryGetProperty("capture_scope", out JsonElement scopeField) && scopeField.GetString() == "step12")
                {
                    string frozenModel = frozen.GetProperty("model").GetString()!;
                    if (values.TryGetValue("--detector-model", out string? supplied) && supplied != frozenModel) throw new ArgumentException("--detector-model conflicts with replay contract.");
                    values["--detector-model"] = frozenModel;
                }
                foreach ((string arg, string field) in new[] { ("--profile", "profile"), ("--nodes", "nodes"), ("--max-traced-lines", "max_traced_lines") })
                    if (!values.ContainsKey(arg) && frozen.TryGetProperty(field, out JsonElement value)) values[arg] = value.ToString();
            }
            string profile = values.GetValueOrDefault("--profile", "canonical"), nodes = values.GetValueOrDefault("--nodes", "checkpoints");
            if (profile is not ("canonical" or "cpu") || nodes is not ("checkpoints" or "all"))
                throw new ArgumentException("Invalid --profile or --nodes.");
            if (new[] { "--input", "--input-bundle", "--replay" }.Count(values.ContainsKey) > 1) throw new ArgumentException("Choose --input, --input-bundle or --replay.");
            string scope = values.GetValueOrDefault("--capture-scope", "ocr"), detector = values.GetValueOrDefault("--detector-model", "tiny"), fixtureSet = values.GetValueOrDefault("--fixture-set", "default");
            if (scope is not ("ocr" or "step12") || detector is not ("tiny" or "small" or "medium") || fixtureSet is not ("default" or "compact" or "expanded"))
                throw new ArgumentException("Invalid capture scope, detector model or fixture set.");
            if (scope == "ocr" && (detector != "tiny" || fixtureSet != "default")) throw new ArgumentException("DetectorModel and FixtureSet require step12 scope; use Model for OCR.");
            if (scope == "step12" && values.ContainsKey("--input-bundle")) throw new ArgumentException("InputBundle is supported for full OCR captures.");
            if (fixtureSet != "default" && (values.ContainsKey("--input") || values.ContainsKey("--replay") || values.ContainsKey("--input-bundle"))) throw new ArgumentException("Fixture set cannot be combined with input/replay.");
            if (scope == "step12" && profile != "canonical") throw new ArgumentException("Step12 references require the canonical profile.");
            string model = values.GetValueOrDefault("--model", "tiny");
            _ = ModelPreset.For(model);
            return new(Path.GetFullPath(Required("--reference-repo")), Path.GetFullPath(Required("--out")),
                Path.GetFullPath(Required("--exporter-source")), values.TryGetValue("--input", out string? input) ? Path.GetFullPath(input) : null,
                values.TryGetValue("--replay", out string? replay) ? Path.GetFullPath(replay) : null,
                profile, nodes, scope == "step12" ? 0 : Number("--max-traced-lines", 3, 0), Number("--count", 20, 1), Number("--warmup", 1, 0), Number("--repeat", 3, 1), scope, detector, fixtureSet, model,
                values.TryGetValue("--input-bundle", out string? inputBundle) ? Path.GetFullPath(inputBundle) : null, replayContract);
        }
    }
}
