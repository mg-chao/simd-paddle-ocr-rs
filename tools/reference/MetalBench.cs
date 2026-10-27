using System.Collections;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Backends.Metal;
using Sdcb.SimdPaddleOCR.OnnxSharp;

// Developer-only adapter. The referenced production C# checkout remains unchanged.
internal static class MetalBench
{
    internal static void Run(string[] args)
    {
        if (args.Length != 6)
            throw new ArgumentException("--metal-bench MODELS PRESET MANIFEST cold|sample|corpus INDEX");
        if (!OperatingSystem.IsMacOS() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
            throw new PlatformNotSupportedException("Physical Apple Silicon required");
        var inputs = JsonSerializer.Deserialize<JsonElement[]>(File.ReadAllBytes(args[3]))!;
        int selected = int.Parse(args[5], CultureInfo.InvariantCulture);
        string mode = args[4];
        if (mode is not ("cold" or "sample" or "corpus") || inputs.Length == 0 || selected < 0 || selected >= inputs.Length)
            throw new ArgumentException("invalid benchmark mode or input index");
        var (det, rec, keys) = args[2] switch
        {
            "tiny" => ("det.onnx", "rec.onnx", "ppocr_keys.txt"),
            "small" => ("small_det.onnx", "small_rec.onnx", "rec_keys.txt"),
            "medium" => ("medium_det.onnx", "medium_rec.onnx", "rec_keys.txt"),
            _ => throw new ArgumentException("preset")
        };
        // Cold includes this read. Corpus reads are outside each inference interval.
        byte[] initialPixels = ReadPixels(inputs[selected]);
        long setupStart = Stopwatch.GetTimestamp();
        using var ocr = PaddleOcrAll.Load(Path.Combine(args[1], det), Path.Combine(args[1], "cls.onnx"),
            Path.Combine(args[1], rec), Path.Combine(args[1], keys), new()
            {
                RecBatchLines = 2, LineWorkerCount = 4, DetIntraOpThreads = 8, PreprocessWorkerCount = 16,
                Detector = new() { Backend = OcrBackend.Metal },
                Classifier = new() { Backend = OcrBackend.Metal },
                Recognizer = new() { Backend = OcrBackend.Metal }
            });
        double setupMs = Stopwatch.GetElapsedTime(setupStart).TotalMilliseconds;
        PipelineProfiler.Enable(false);
        InferenceSession.EnableProfiling(false);
        var saved = new Dictionary<string, string>();
        int calls = mode == "cold" ? 1 : mode == "sample" ? 2 : inputs.Length * 2;
        for (int call = 0; call < calls; call++)
        {
            int i = mode == "corpus" ? call % inputs.Length : selected;
            var entry = inputs[i];
            string id = entry.GetProperty("id").GetString()!;
            string phase = mode == "corpus" ? (call < inputs.Length ? "first_shape" : "warm") : call == 0 ? "cold" : "warm";
            byte[] pixels = i == selected && (mode != "corpus" || call == 0) ? initialPixels : ReadPixels(entry);
            int width = entry.GetProperty("width").GetInt32(), height = entry.GetProperty("height").GetInt32();
            long start = Stopwatch.GetTimestamp();
            var result = ocr.Run(pixels, width, height, false, 0, ImagePixelFormat.Rgb24);
            double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            // Signal BEFORE reflection, output serialization, and correctness checks.
            Console.WriteLine($"RESULT_READY {id} {phase} {ms.ToString("R", CultureInfo.InvariantCulture)}");
            Console.Out.Flush();
            CheckGpu(ocr);
            var output = Canonical(result);
            string encoded = JsonSerializer.Serialize(output);
            if (saved.TryGetValue(id, out string? previous) && previous != encoded)
                throw new InvalidOperationException("nondeterministic OCR output: " + id);
            saved[id] = encoded;
            Console.WriteLine(JsonSerializer.Serialize(new { kind = "image", id, phase, ms, output }));
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            kind = "summary", implementation = "csharp", preset = args[2], mode, calls,
            backend = "Metal", adapter = MetalBackend.TryGetDevice()?.Name, backend_verified = true,
            precision = "mixed_f16", setup_ms = setupMs, framework = RuntimeInformation.FrameworkDescription,
            timing_contract = "RGB to owned result; parent launch-to-first-RESULT_READY includes process, raw input, models, Metal and inference; validation and serialization excluded"
        }));
    }

    private static byte[] ReadPixels(JsonElement input)
    {
        var pixels = File.ReadAllBytes(input.GetProperty("rgb").GetString()!);
        if (pixels.Length != checked(input.GetProperty("width").GetInt32() * input.GetProperty("height").GetInt32() * 3))
            throw new ArgumentException("RGB byte length mismatch");
        return pixels;
    }

    private static void CheckGpu(PaddleOcrAll ocr)
    {
        foreach (string stage in new[] { "detector", "classifier", "recognizer" })
        {
            var owner = typeof(PaddleOcrAll).GetField("_" + stage, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(ocr);
            if (owner is null) throw new InvalidOperationException("missing " + stage);
            var sessions = ((IEnumerable)owner.GetType().GetField("_sessions", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(owner)!).Cast<IOcrSession>().ToArray();
            // Empty pages legitimately leave CLS/REC uninvoked.
            if ((stage == "detector" && sessions.Length == 0) || sessions.Any(s => !s.GpuAlive || s.GetType() != typeof(MetalSession)))
                throw new InvalidOperationException("C# Metal fallback: " + stage);
        }
    }

    private static object Canonical(PaddleOcrResult result) => new
    {
        text = result.Text, detected_count = result.DetectedCount,
        detector_resized_width = result.DetectorResizedWidth, detector_resized_height = result.DetectorResizedHeight,
        lines = result.Lines.Select(line => new
        {
            text = line.Text, detection = new
            {
                points = Enumerable.Range(0, 4).Select(i => new[] { line.Box[i].X, line.Box[i].Y }), score = line.Box.Score
            },
            recognition_score = line.RecognitionScore, classification_score = line.ClassificationScore,
            classification_label = line.ClassificationLabel, applied_rotation_degrees = line.AppliedRotationDegrees,
            emitted_count = line.EmittedCount, recognition_content_width = line.RecognitionContentWidth,
            recognition_tensor_width = line.RecognitionTensorWidth, recognition_time_steps = line.RecognitionTimeSteps,
            ctc_spans = line.CtcSpans?.Select(s => new { text = s.Text, score = s.Score, start_column = s.StartColumn, end_column = s.EndColumn })
        })
    };
}
