using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Backends.Metal;

// Developer-only comparison adapter: production remains pure Rust/Metal.
internal static class OcrBench
{
    internal static void Run(string[] args)
    {
        if (args.Length != 9) throw new ArgumentException("--ocr-bench MODELS PRESET cpu|gpu RGB WIDTH HEIGHT REQUESTS BATCH");
        string dir = args[1], preset = args[2], mode = args[3];
        int width = int.Parse(args[5]), height = int.Parse(args[6]), requests = int.Parse(args[7]), batch = int.Parse(args[8]);
        if (requests is < 1 or > 16 || batch is < 1 or > 64) throw new ArgumentException("request/batch bounds");
        OcrBackend backend = mode switch { "cpu" => OcrBackend.Cpu, "gpu" => OcrBackend.Metal, _ => throw new ArgumentException("backend") };
        (string det,string rec,string keys) = preset switch {
            "tiny" => ("det.onnx","rec.onnx","ppocr_keys.txt"),
            "small" => ("small_det.onnx","small_rec.onnx","rec_keys.txt"),
            "medium" => ("medium_det.onnx","medium_rec.onnx","rec_keys.txt"),
            _ => throw new ArgumentException("preset")
        };
        byte[] pixels = File.ReadAllBytes(args[4]);
        if (pixels.Length != checked(width * height * 3)) throw new ArgumentException("RGB size mismatch");
        bool capture = Environment.GetEnvironmentVariable("SIMD_OCR_BENCH_CAPTURE") == "1";
        bool profile = Environment.GetEnvironmentVariable("SIMD_OCR_BENCH_PROFILE") == "1";
        int measurements = int.Parse(Environment.GetEnvironmentVariable("SIMD_OCR_BENCH_MEASUREMENTS") ?? "3");
        if (measurements is < 0 or > 3) throw new ArgumentException("measurement bounds");
        var clock = Stopwatch.StartNew();
        using var ocr = PaddleOcrAll.Load(Path.Combine(dir,det),Path.Combine(dir,"cls.onnx"),Path.Combine(dir,rec),Path.Combine(dir,keys),new() {
            RecBatchLines = batch, LineWorkerCount = 4, DetIntraOpThreads = 8, PreprocessWorkerCount = 16,
            Detector = new() { Backend = backend }, Classifier = new() { Backend = backend }, Recognizer = new() { Backend = backend }
        });
        double startup = clock.Elapsed.TotalMilliseconds;
        (double, PaddleOcrResult[]) Execute()
        {
            var results = new PaddleOcrResult[requests]; var complete = new double[requests]; var failures = new Exception?[requests];
            long start = Stopwatch.GetTimestamp();
            var threads = Enumerable.Range(0,requests).Select(i => new Thread(() => {
                try { results[i] = ocr.Run(pixels,width,height,capture,0,ImagePixelFormat.Rgb24); complete[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds; }
                catch (Exception e) { failures[i] = e; }
            })).ToArray();
            foreach (var thread in threads) thread.Start();
            foreach (var thread in threads) thread.Join();
            if (failures.Any(e => e is not null)) throw new AggregateException(failures.OfType<Exception>());
            return (complete.Max(),results);
        }
        PipelineProfiler.Enable(false); InferenceSession.EnableProfiling(false);
        var (cold, outputs) = Execute();
        var canonical = outputs.Select(Canonical).ToArray();
        var errors = new List<string>();
        if (Environment.GetEnvironmentVariable("SIMD_OCR_BENCH_REFERENCE") is string reference) {
            var expected = JsonNode.Parse(File.ReadAllText(reference))!;
            foreach (var result in canonical) Check(result,expected,capture,mode == "gpu",errors);
        }
        object[] Sessions() => new[] { "detector","classifier","recognizer" }.SelectMany(stage => {
            object? owner = typeof(PaddleOcrAll).GetField("_"+stage,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(ocr);
            if (owner is null) return Array.Empty<object>();
            var sessions = (IEnumerable)owner.GetType().GetField("_sessions",BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(owner)!;
            return sessions.Cast<IOcrSession>().Select(s => (object)new { stage, type = s.GetType().FullName, gpu_alive = s.GpuAlive });
        }).ToArray();
        bool GpuAlive() => new[] { "detector","classifier","recognizer" }.All(stage => {
            object? owner = typeof(PaddleOcrAll).GetField("_"+stage,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(ocr);
            if (owner is null) return true;
            var sessions = ((IEnumerable)owner.GetType().GetField("_sessions",BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(owner)!).Cast<IOcrSession>().ToArray();
            // Empty pages may legitimately never invoke CLS/REC.
            string expected = "Backends.Metal";
            return (stage != "detector" || sessions.Length > 0) && sessions.All(s => s.GetType().FullName!.Contains(expected) && s.GpuAlive);
        });
        if (mode == "gpu" && !GpuAlive()) errors.Add("C# GPU fell back to CPU");
        var measured = new List<double>();
        for (int i = 0; i < measurements && errors.Count == 0; i++) {
            if (clock.Elapsed.TotalSeconds >= 60) throw new TimeoutException("60 second invocation budget");
            var (elapsed, actual) = Execute();
            if (!actual.Select(Canonical).Zip(canonical).All(pair => JsonNode.DeepEquals(pair.First,pair.Second))) errors.Add("nondeterministic outputs");
            if (mode == "gpu" && !GpuAlive()) errors.Add("C# GPU fell back to CPU");
            measured.Add(elapsed);
        }
        object? profiles = null;
        if (profile && errors.Count == 0) {
            PipelineProfiler.Enable(true);
            var (_, actual) = Execute();
            if (!actual.Select(Canonical).Zip(canonical).All(pair => JsonNode.DeepEquals(pair.First,pair.Second))) errors.Add("profiling changes output");
            profiles = PipelineProfiler.Snapshot().Select((s,i) => new { stage = PipelineProfiler.StageNames[i], milliseconds = s.Milliseconds, calls = s.Calls }).ToArray();
            PipelineProfiler.Enable(false);
        }
        double? median = measured.Count == 0 ? null : measured.Order().ElementAt(measured.Count/2);
        Console.WriteLine(JsonSerializer.Serialize(new {
            implementation = "csharp", pipeline = mode, preset, requests, batch, capture,
            adapter = mode == "gpu" ? MetalBackend.TryGetDevice()?.Name : null,
            backend = backend.ToString(),
            framework = RuntimeInformation.FrameworkDescription, precision = mode == "gpu" ? "mixed_f16" : "f32",
            cpu_options = new { line_workers = 4, detector_threads = 8, preprocess_workers = 16 },
            timing_contract = "group start through last inference return; excludes image decoding, validation, serialization and profile calls; C# public results are owned",
            startup_ms = startup, cold_run_ms = cold, warm_group_ms = measured, median_group_ms = errors.Count == 0 ? median : null,
            images_per_second = errors.Count == 0 && median is > 0 ? requests * 1000 / median : null,
            qualified = errors.Count == 0, errors, outputs = canonical, sessions = Sessions(), profiles,
            reference_scope = Environment.GetEnvironmentVariable("SIMD_OCR_BENCH_REFERENCE_SCOPE") ?? "external-reference",
            retained_device_bytes = (long?)null, peak_device_bytes = (long?)null
        },new JsonSerializerOptions { WriteIndented = true }));
    }
    static JsonNode Canonical(PaddleOcrResult r) => JsonSerializer.SerializeToNode(new {
        text = r.Text, detected_count = r.DetectedCount, detector_resized_width = r.DetectorResizedWidth, detector_resized_height = r.DetectorResizedHeight,
        lines = r.Lines.Select(l => new {
            text = l.Text, detection = new { points = Enumerable.Range(0,4).Select(i => new[] { l.Box[i].X,l.Box[i].Y }), score = l.Box.Score },
            recognition_score = l.RecognitionScore, classification_score = l.ClassificationScore, classification_label = l.ClassificationLabel,
            applied_rotation_degrees = l.AppliedRotationDegrees, emitted_count = l.EmittedCount,
            recognition_content_width = l.RecognitionContentWidth, recognition_tensor_width = l.RecognitionTensorWidth, recognition_time_steps = l.RecognitionTimeSteps,
            ctc_spans = l.CtcSpans?.Select(s => new { text = s.Text, score = s.Score, start_column = s.StartColumn, end_column = s.EndColumn })
        })
    })!;
    static void Check(JsonNode a,JsonNode b,bool capture,bool mixed,List<string> errors)
    {
        void Same(JsonNode? a,JsonNode? b,string field) { if (!JsonNode.DeepEquals(a,b)) errors.Add(field); }
        void Close(JsonNode? a,JsonNode? b,string field,bool coordinate = false) {
            double x = a!.GetValue<double>(), y = b!.GetValue<double>();
            double tolerance = coordinate ? .5 : mixed ? .05 + .01 * Math.Abs(y) : .001 + .001 * Math.Abs(y);
            if (!double.IsFinite(x) || !double.IsFinite(y) || Math.Abs(x-y) > tolerance) errors.Add($"{field}: {x} vs {y}");
        }
        foreach (string f in new[] {"text","detected_count","detector_resized_width","detector_resized_height"}) Same(a[f],b[f],f);
        var aa = a["lines"]!.AsArray(); var bb = b["lines"]!.AsArray();
        if (aa.Count != bb.Count) errors.Add("line count");
        for (int i=0;i<Math.Min(aa.Count,bb.Count);i++) {
            var x=aa[i]!;var y=bb[i]!;
            foreach (string f in new[] {"text","classification_label","applied_rotation_degrees","recognition_content_width","recognition_tensor_width","recognition_time_steps","emitted_count"}) Same(x[f],y[f],$"line {i}: {f}");
            foreach (string f in new[] {"recognition_score","classification_score"}) Close(x[f],y[f],$"line {i}: {f}");
            Close(x["detection"]!["score"],y["detection"]!["score"],$"line {i}: detector score");
            for (int p=0;p<4;p++) for(int d=0;d<2;d++) Close(x["detection"]!["points"]![p]![d],y["detection"]!["points"]![p]![d],$"line {i}: point {p}/{d}",true);
            if (capture) {
                if (x["ctc_spans"] is not JsonArray xs || y["ctc_spans"] is not JsonArray ys) { errors.Add("missing CTC spans");continue; }
                if (xs.Count!=ys.Count) errors.Add("CTC count");
                foreach(var pair in xs.Zip(ys)) { foreach(string f in new[]{"text","start_column","end_column"}) Same(pair.First![f],pair.Second![f],$"line {i}: CTC {f}");Close(pair.First!["score"],pair.Second!["score"],$"line {i}: CTC score"); }
            }
        }
    }
}
