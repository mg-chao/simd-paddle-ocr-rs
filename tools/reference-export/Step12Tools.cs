using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Canonical references through oriented crops; recognition is never executed.</summary>
internal static class Step12Tools
{
    internal static PaddleOcrDetectorOptions DetectorOptions(string variant) => new()
    {
        Backend = OcrBackend.Cpu, BoxThreshold = variant == "tiny" ? 0.4f : 0.45f,
        MaxPooledSessions = 1,
    };

    internal static void RequireCanonical()
    {
        foreach ((string key, string value) in new[] {
            ("DOTNET_PROCESSOR_COUNT", "1"), ("DOTNET_EnableHWIntrinsic", "0"),
            ("PPOCR_NHWC", "0"), ("SIMD_OCR_NOFUSE", "1"), ("PPOCR_DET_THREADS", "1") })
            if (Environment.GetEnvironmentVariable(key) != value)
                throw new InvalidOperationException($"{key} must be {value}; use the canonical launcher.");
        if (Environment.ProcessorCount != 1 || Vector.IsHardwareAccelerated)
            throw new InvalidOperationException("Canonical export requires serial scalar execution.");
        foreach (string name in new[] { "PPOCR_DUMP_CONV", "PPOCR_DUMP_NODES", "PPOCR_DUMP_PLAN", "PPOCR_DUMP_GRAPH" })
            if (Environment.GetEnvironmentVariable(name) is not null) throw new InvalidOperationException($"{name} must be absent.");
    }

    public static void Export(Program.Settings settings)
    {
        RequireCanonical();
        List<Program.Fixture> fixtures = Fixtures(settings);
        BundleWriter writer = new(settings.Output);
        string modelRoot = Path.Combine(settings.ReferenceRepo, "models");
        string detFile = settings.DetectorModel == "tiny" ? "det.onnx" : $"{settings.DetectorModel}_det.onnx";
        string detectorPath = Path.Combine(modelRoot, detFile), classifierPath = Path.Combine(modelRoot, "cls.onnx");
        if (settings.Replay is not null)
        {
            using JsonDocument replay = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(settings.Replay, "manifest.json")));
            string variant = replay.RootElement.GetProperty("contract").GetProperty("model").GetString()!;
            if (variant != settings.DetectorModel) throw new InvalidDataException("Step12 replay detector conflicts with the frozen contract.");
            JsonElement[] models = replay.RootElement.GetProperty("models").EnumerateArray().ToArray();
            if (models.Length != 2) throw new InvalidDataException("Step12 replay requires exactly DET and CLS assets.");
            string Source(string id)
            {
                JsonElement asset = models.Single(asset => asset.GetProperty("id").GetString() == id);
                _ = Program.ReadBundleFile(settings.Replay, asset);
                return BundleWriter.DestinationPath(settings.Replay, asset.GetProperty("path").GetString()!);
            }
            detectorPath = Source("det"); classifierPath = Source("cls");
        }
        ModelAsset[] assets = [
            writer.CopyModel("det", detectorPath, "models/det.onnx"),
            writer.CopyModel("cls", classifierPath, "models/cls.onnx") ];
        object provenance = Provenance.Capture(settings.ReferenceRepo, settings.ExporterSource, writer.RootPath);
        Dictionary<string, ModelInspectionSnapshot> inspected = [];
        Model Load(string id)
        {
            using FileStream stream = File.OpenRead(Path.Combine(writer.RootPath, $"models/{id}.onnx"));
            return Model.Load(stream, snapshot => { if (snapshot.Stage == "normalized") inspected.Add(id, snapshot); });
        }
        using Model det = Load("det");
        using Model cls = Load("cls");
        PaddleOcrDetectorOptions options = DetectorOptions(settings.DetectorModel);
        using PaddleOcrDetector detector = new(det, options, 1);
        using PaddleOcrClassifier classifier = new(cls, new() { Backend = OcrBackend.Cpu, MaxPooledSessions = 1 });
        List<CaseEntry> cases = [];
        foreach (Program.Fixture fixture in fixtures)
        {
            List<Artifact> artifacts = [];
            string directory = $"cases/{fixture.Id}";
            artifacts.Add(writer.WriteJson("input.metadata", $"{directory}/input.json", new {
                fixture.Width, fixture.Height, fixture.Stride, pixelFormat = "bgr24", fixture.Origin }));
            artifacts.Add(writer.WriteBytes("input.pixels", $"{directory}/input.bgr.bin", fixture.Pixels, [fixture.Height, fixture.Stride], "bgr24_rows"));
            PaddleOcrDetectionBox[] boxes = Program.CaptureDetector(writer, artifacts, directory, fixture, det, settings, options);
            PaddleOcrDetectionResult publicDetection = detector.Detect(fixture.Pixels, fixture.Width, fixture.Height, fixture.Stride);
            if (!boxes.AsSpan().SequenceEqual(publicDetection.Boxes))
                throw new InvalidDataException($"Diagnostic/public detector differs for {fixture.Id}.");
            List<object> lines = [];
            for (int index = 0; index < boxes.Length; index++)
            {
                string stage = $"line.{index:D3}";
                byte[] crop = PPOCRCrop.Extract(fixture.Pixels, fixture.Width, fixture.Height, fixture.Stride, boxes[index], out int width, out int height);
                artifacts.Add(writer.WriteBytes($"{stage}.crop", $"{directory}/{stage}/crop.bgr.bin", crop, [height, width, 3], "bgr24"));
                PaddleOcrClassificationResult classification = classifier.Classify(crop, width, height);
                Program.CaptureClassifier(writer, artifacts, directory, stage, crop, width, height, cls, settings, classification);
                int rotation = classification.Label == 1 && classification.Score > 0 ? 180 : 0;
                if (rotation == 180) PPOCRCrop.Rotate180(crop, width, height);
                artifacts.Add(writer.WriteBytes($"{stage}.rec.pixels", $"{directory}/{stage}/rec.bgr.bin", crop, [height, width, 3], "bgr24"));
                lines.Add(new { box = Program.BoxJson(boxes[index]), classificationScore = classification.Score,
                    classificationLabel = classification.Label, appliedRotationDegrees = rotation, width, height });
            }
            artifacts.Add(writer.WriteJson("step12.public", $"{directory}/step12.json", new {
                detectedCount = boxes.Length, detectorResizedWidth = publicDetection.ResizedWidth,
                detectorResizedHeight = publicDetection.ResizedHeight, publicDetection.WidthRatio, publicDetection.HeightRatio, lines }));
            cases.Add(new(fixture.Id, "step12", artifacts));
            Console.WriteLine($"captured {settings.DetectorModel}/{fixture.Id}: {boxes.Length} complete lines, {artifacts.Count} artifacts");
        }
        var inventory = new[] { "det", "cls" }.Select(id => new {
            id, inputs = inspected[id].Inputs, outputs = inspected[id].Outputs,
            nodes = inspected[id].Nodes.Select((node, index) => new {
                index, node.Name, node.OpType, node.Domain, node.Opset, inputs = node.Inputs, outputs = node.Outputs }).ToArray() }).ToArray();
        Artifact inventoryFile = writer.WriteJson("graph_inventory", "graph-inventory.json", inventory);
        writer.WriteManifest(new {
            schemaVersion = 1,
            contract = new { model = settings.DetectorModel, captureScope = "step12", profile = "canonical", backend = "cpu",
                pixelFormat = "bgr24", logicalInputLayout = "nchw", tensorEncoding = "ieee754_f32_little_endian",
                nodes = settings.Nodes, maxTracedLines = 0, inputOrigin = "frozen_bgr_pixels",
                options = new { detector = options, useDirectionClassification = true, classifierThreshold = 0f,
                    maxCropPixels = 16_000_000L, maxImagePixels = 40_000_000L },
                publicClassifierSemantics = "C# Classify: compatible bias-first head/logit argmax, separate from dense Softmax probabilities; strict score > threshold rotation.",
                scope = "Detection, ordered boxes, perspective crops, classifier and oriented recognition pixels; no recognition graph or text." },
            models = assets, cases, provenance,
            auxiliaryFiles = ImportTools.ProvenanceFiles((ReferenceProvenance)provenance).Concat(new[] { ImportTools.Auxiliary(inventoryFile) }).ToArray() });
        Console.WriteLine($"saved complete step12 reference: {writer.RootPath}");
    }

    internal static List<Program.Fixture> Fixtures(Program.Settings settings)
    {
        if (settings.FixtureSet == "default") return Program.LoadFixtures(settings);
        List<Program.Fixture> result = [ Pattern("landscape", 64, 32), Pattern("portrait", 32, 64),
            new("blank", Enumerable.Repeat((byte)255, 32 * 32 * 3).ToArray(), 32, 32, 96, "white 32x32", null, ""),
            new("tiny", [31, 127, 223], 1, 1, 3, "one BGR pixel", null, "") ];
        if (settings.FixtureSet == "expanded")
        {
            result.AddRange(Program.LoadFixtures(settings));
            // Padded rectangular sources exercise byte strides and independent axis mapping.
            result.Add(Pattern("stride-landscape", 511, 257, 7));
            result.Add(Pattern("stride-portrait", 257, 511, 11));
            result.Add(Pattern("wide", 1025, 129));
            result.Add(Pattern("tall", 129, 1025));
            result.Add(Pattern("rounding", 47, 49));
        }
        return result.GroupBy(fixture => fixture.Id, StringComparer.Ordinal).Select(group => group.First()).ToList();
    }

    private static Program.Fixture Pattern(string id, int width, int height, int padding = 0)
    {
        int stride = checked(width * 3 + padding);
        byte[] pixels = Enumerable.Repeat((byte)173, stride * height).ToArray();
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) for (int c = 0; c < 3; c++)
            pixels[y * stride + x * 3 + c] = (byte)((x * 17 + y * 29 + c * 67) & 255);
        return new(id, pixels, width, height, stride, "deterministic x*17+y*29+channel*67 mod256; padding=173", null, "");
    }

    public static int Benchmark(string[] args) => Step12Benchmark.Run(args);
}
