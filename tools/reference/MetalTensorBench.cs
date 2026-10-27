using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Backends.Metal;

// Bounded developer diagnostic: one correctness/warmup call and three
// unprofiled calls, including logical input upload and F32 output readback.
internal static class MetalTensorBench
{
    internal static void Run(string[] args)
    {
        if (args.Length != 7) throw new ArgumentException("MODEL INPUT.f32 OUTPUT.f32 N C H W");
        using var model = Model.Load(File.ReadAllBytes(args[0]));
        var compiled = new CompiledModel(model, intraOpThreads: 1);
        int[] shape = args[3..].Select(int.Parse).ToArray();
        float[] input = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(args[1])).ToArray();
        if (input.Length != shape.Aggregate(1, (a, b) => checked(a * b)))
            throw new ArgumentException("input size mismatch");
        using var cpu = new InferenceSession(compiled);
        cpu.Reshape(shape);
        float[] expected = cpu.Run(input).ToArray();
        var device = MetalBackend.TryGetDevice() ?? throw new NotSupportedException("physical Metal device required");
        using var gpu = new MetalDetGraph(device, compiled);
        float[] native = gpu.Run(shape, input).ToArray();
        // The reference's internal runner returns its physical NHWC slab for
        // rank-four feature maps (sigmoid_out only converts element type).
        // Canonicalize outside the measured calls for the CPU oracle.
        int[] outputShape = cpu.OutputShape.Dimensions;
        if (outputShape.Length != 4) throw new ArgumentException("diagnostic requires a rank-four output");
        float[] actual = new float[native.Length];
        int channels = outputShape[1], pixels = outputShape[2] * outputShape[3];
        for (int n = 0; n < outputShape[0]; ++n)
            for (int c = 0; c < channels; ++c)
                for (int p = 0; p < pixels; ++p)
                    actual[(n * channels + c) * pixels + p] = native[(n * pixels + p) * channels + c];
        if (actual.Length != expected.Length) throw new InvalidOperationException("output size mismatch");
        float maxAbs = 0;
        int failures = 0;
        for (int i = 0; i < actual.Length; ++i)
        {
            float error = MathF.Abs(actual[i] - expected[i]);
            maxAbs = MathF.Max(maxAbs, error);
            if (!float.IsFinite(actual[i]) || !float.IsFinite(expected[i]) || error > .05f + .01f * MathF.Abs(expected[i])) ++failures;
        }
        File.WriteAllBytes(args[2], MemoryMarshal.AsBytes(actual.AsSpan()).ToArray());
        var measurements = new List<object>();
        if (failures == 0)
            for (int i = 0; i < 3; ++i)
            {
                long start = Stopwatch.GetTimestamp();
                gpu.Run(shape, input);
                measurements.Add(new { wall_ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds });
            }
        Console.WriteLine(JsonSerializer.Serialize(new { shape, output_shape = outputShape, adapter = device.Name, backend = "Metal", precision = "mixed_f16", output_layout = "NHWC (oracle conversion excluded)", profiled = false, max_abs = maxAbs, failures, measurements }));
        if (failures != 0) throw new InvalidOperationException("GPU output exceeds the mixed-precision gate");
    }
}
