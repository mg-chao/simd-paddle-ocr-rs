using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

// DET-only production node counters: separate from stage and official timing.
// Node slots are static and keyed by graph index, so combining DET/CLS/REC in
// one pass would merge unrelated operators. Restricting to DET avoids that.
static class NodeProbe
{
    public static void Run(string detectorPath, byte[] pixels, int width, int height,
        string modelId, string fixtureId, string affinity, PaddleOcrDetectorOptions options)
    {
        using Model model = Model.Load(File.OpenRead(detectorPath));
        using PaddleOcrDetector detector = new(model, options, 1);
        PaddleOcrDetectionResult control = detector.Detect(pixels, width, height, width * 3, ImagePixelFormat.Bgr24);
        PipelineProfiler.Enable(false);
        InferenceSession.EnableProfiling(true);
        Stopwatch timer = Stopwatch.StartNew();
        PaddleOcrDetectionResult result = detector.Detect(pixels, width, height, width * 3, ImagePixelFormat.Bgr24);
        double profiledMs = timer.Elapsed.TotalMilliseconds;
        var counters = InferenceSession.NodeProfileSnapshot();
        InferenceSession.EnableProfiling(false);
        if (result.ResizedWidth != control.ResizedWidth || result.ResizedHeight != control.ResizedHeight || !result.Boxes.SequenceEqual(control.Boxes))
            throw new InvalidDataException("node-profiled detector output differs from warm-up");
        // Inspect retained request metadata after all timed work; no observer or
        // reference session is ever inserted into the execution path.
        IEnumerable pool = (IEnumerable)typeof(PaddleOcrDetector).GetField("_sessions", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(detector)!;
        object request = pool.Cast<object>().Single();
        Array tensors = (Array)typeof(InferenceSession).GetField("_tensors", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(request)!;
        int[] Shape(uint id)
        {
            object tensor = tensors.GetValue((int)id)!;
            return (int[])tensor.GetType().GetProperty("Shape")!.GetValue(tensor)!;
        }
        var top = counters.Select((counter, index) => new { counter, index })
            .Where(item => item.counter.Calls > 0 && item.index < model.Nodes.Length)
            .OrderByDescending(item => item.counter.Ticks).Take(30).Select(item =>
            {
                NodeRecord node = model.Nodes[item.index];
                return new
                {
                    index = item.index, node.Name, operation = node.Operator.ToString(),
                    calls = item.counter.Calls, milliseconds = item.counter.Ticks * 1000.0 / Stopwatch.Frequency,
                    input_shapes = node.Inputs.Where(id => id != uint.MaxValue).Select(Shape).ToArray(),
                    output_shapes = node.Outputs.Select(Shape).ToArray(),
                };
            }).ToArray();
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            diagnostic = true, official_benchmark = false, production_fusion = true,
            mode = "detector_production_nodes", model = modelId, fixture = fixtureId, threads = 1, affinity,
            runtime = Environment.Version.ToString(), warmup_runs = 1, measured_runs = 1,
            profiled_detector_total_ms = profiledMs, output_matches_warmup = true,
            node_counter_capacity = counters.Length, model_nodes = model.Nodes.Length,
            note = "Only DET runs under production node instrumentation; fused/skipped nodes retain production behavior. Counter capacity is512. Node timings are diagnostic and not official performance evidence.",
            top_nodes = top,
        }, new JsonSerializerOptions { WriteIndented = true }));
    }
}
