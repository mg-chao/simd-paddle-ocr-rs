using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using Sdcb.SimdPaddleOCR.Kernels;

// Focused diagnosis, not official OCR throughput: selected fixed kernel shapes,
// one warm-up invocation, three samples, fixed iterations, ten seconds total.
static class KernelProbe
{
    public static void Run(bool recognition = false)
    {
        if (!OperatingSystem.IsWindows() || !Environment.Is64BitProcess || Environment.ProcessorCount != 1)
            throw new InvalidOperationException("kernel diagnosis requires Windows x64 and DOTNET_PROCESSOR_COUNT=1");
        if (!Avx2.IsSupported || !Fma.IsSupported || Avx512F.IsSupported)
            throw new InvalidOperationException("this matched probe qualifies the current AVX2/FMA host only");
        ulong available = unchecked((ulong)(long)Process.GetCurrentProcess().ProcessorAffinity);
        ulong affinity = available & unchecked(0UL - available);
        Process.GetCurrentProcess().ProcessorAffinity = unchecked((IntPtr)(long)affinity);
        Stopwatch budget = Stopwatch.StartNew();
        List<object> results = [];
        if (recognition)
        {
            results.Add(RecognitionPointwise(96, 384, 768, budget));
            results.Add(RecognitionPointwise(960, 512, 1024, budget));
        }
        else
        {
            foreach ((int h, int w) in new[] { (19, 30), (76, 120), (152, 240) })
                results.Add(Pointwise(h, w, budget));
            results.Add(Depthwise(budget));
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            diagnostic = true, official_benchmark = false, runtime = Environment.Version.ToString(),
            threads = 1, affinity = affinity.ToString("x"), warmup_calls_per_case = 1,
            samples_per_case = 3, budget_seconds = 10, elapsed_ms = budget.Elapsed.TotalMilliseconds,
            results,
        }, new JsonSerializerOptions { WriteIndented = true }));
    }
    static void CheckBudget(Stopwatch budget)
    {
        if (budget.Elapsed.TotalSeconds >= 10) throw new TimeoutException("fixed ten-second kernel diagnostic budget exceeded");
    }
    static object RecognitionPointwise(int rows, int inner, int columns, Stopwatch budget)
    {
        float[] input = Input(rows * inner), matrix = Weights(inner * columns), bias = Bias(columns), output = new float[rows * columns];
        float[] weights = new float[matrix.Length];
        for (int k = 0; k < inner; k++)
            for (int c = 0; c < columns; c++) weights[c * inner + k] = matrix[k * columns + c];
        float[] packed = Nhwc.PackDense(weights, columns, inner, 1);
        void Call() => Nhwc.Pointwise(input, packed, bias, output, rows, inner, columns, [], NhwcActivation.None, 0, 0, 1);
        CheckBudget(budget); Call();
        List<double> samples = [];
        for (int run = 0; run < 3; run++)
        {
            CheckBudget(budget); Stopwatch timer = Stopwatch.StartNew(); Call(); samples.Add(timer.Elapsed.TotalMilliseconds);
        }
        foreach (int row in new[] { 0, 5, rows - 1 })
            foreach (int c in new[] { 0, 7, columns - 1 })
            {
                float expected = bias[c];
                for (int k = 0; k < inner; k++) expected = MathF.FusedMultiplyAdd(input[row * inner + k], matrix[k * columns + c], expected);
                if (BitConverter.SingleToInt32Bits(output[row * columns + c]) != BitConverter.SingleToInt32Bits(expected))
                    throw new InvalidDataException("recognition pointwise ordered scalar check failed");
            }
        CheckBudget(budget);
        return new { kernel = "nhwc_recognition_pointwise", rows, inner, columns, calls_per_sample = 1, samples_ms_per_call = samples, median_ms = samples.Order().ElementAt(1) };
    }
    static float[] Input(int length) => Enumerable.Range(0, length).Select(i => ((i * 37 % 43) - 21f) / 17f).ToArray();
    static float[] Weights(int length) => Enumerable.Range(0, length).Select(i => ((i * 19 % 29) - 14f) / 13f).ToArray();
    static float[] Bias(int length) => Enumerable.Range(0, length).Select(i => i / 19f - 0.3f).ToArray();
    static float[] DepthValues(int length, int seed) => Enumerable.Range(0, length).Select(i => ((i * 37 + seed * 17) % 101) / 101f - 0.5f).ToArray();
    static float[] PackPointwise(float[] weights, int cin, int cout, int tile)
    {
        float[] packed = new float[cin * cout];
        for (int block = 0; block < cout / tile; block++)
            for (int ci = 0; ci < cin; ci++)
                for (int lane = 0; lane < tile; lane++)
                    packed[(block * cin + ci) * tile + lane] = weights[(block * tile + lane) * cin + ci];
        return packed;
    }
    static object Pointwise(int h, int w, Stopwatch budget)
    {
        const int cin = 96, cout = 24, calls = 4;
        int plane = h * w;
        float[] input = Input(cin * plane), weights = Weights(cin * cout), bias = Bias(cout), output = new float[cout * plane];
        float[] packed4 = PackPointwise(weights, cin, cout, 4), packed8 = PackPointwise(weights, cin, cout, 8);
        void Call()
        {
            if (!Conv1x1.TryPacked(input, packed4, bias, output, 1, cin, h, w, cout, 1, packedOc8: packed8))
                throw new InvalidOperationException("pointwise production dispatcher rejected probe");
        }
        CheckBudget(budget); Call();
        List<double> samples = [];
        for (int run = 0; run < 3; run++)
        {
            CheckBudget(budget); Stopwatch timer = Stopwatch.StartNew();
            for (int iteration = 0; iteration < calls; iteration++) Call();
            samples.Add(timer.Elapsed.TotalMilliseconds / calls);
        }
        // Sparse independent scalar checks are outside timings. Check first/last
        // channels and both aligned/interior and tail spatial positions.
        foreach (int co in new[] { 0, 3, 8, 23 })
            foreach (int spatial in new[] { 0, Math.Min(17, plane - 1), plane - 1 })
            {
                float expected = bias[co];
                for (int ci = 0; ci < cin; ci++) expected = MathF.FusedMultiplyAdd(input[ci * plane + spatial], weights[co * cin + ci], expected);
                if (MathF.Abs(output[co * plane + spatial] - expected) > 1e-4f + 1e-4f * MathF.Abs(expected))
                    throw new InvalidDataException("pointwise scalar check failed");
            }
        CheckBudget(budget);
        return new { kernel = "nchw_pointwise_96_to_24", height = h, width = w, calls_per_sample = calls, samples_ms_per_call = samples, median_ms = samples.Order().ElementAt(1), finite_output = output.All(float.IsFinite) };
    }
    static object Depthwise(Stopwatch budget)
    {
        const int channels = 96, h = 152, w = 240, kernel = 7;
        float[] input = DepthValues(channels * h * w, 53), weights = DepthValues(channels * kernel * kernel, 59), bias = DepthValues(channels, 61), output = new float[channels * h * w];
        float[] packed = Nhwc.PackDepthwise(weights, channels, kernel * kernel);
        void Call() => Nhwc.Depthwise(input, packed, bias, output, 1, channels, h, w, h, w,
            kernel, kernel, 1, 1, 3, 3, [], NhwcActivation.None, 0, 0, 1);
        CheckBudget(budget); Call();
        List<double> samples = [];
        for (int run = 0; run < 3; run++)
        {
            CheckBudget(budget); Stopwatch timer = Stopwatch.StartNew(); Call(); samples.Add(timer.Elapsed.TotalMilliseconds);
        }
        foreach (int channel in new[] { 0, 7, 31, 95 })
            foreach ((int y, int x) in new[] { (0, 0), (76, 120), (151, 239) })
            {
                float expected = bias[channel];
                for (int ky = 0; ky < kernel; ky++)
                    for (int kx = 0; kx < kernel; kx++)
                    {
                        int iy = y + ky - 3, ix = x + kx - 3;
                        if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                        expected = MathF.FusedMultiplyAdd(input[(iy * w + ix) * channels + channel], weights[(channel * kernel + ky) * kernel + kx], expected);
                    }
                if (MathF.Abs(output[(y * w + x) * channels + channel] - expected) > 1e-4f + 1e-4f * MathF.Abs(expected))
                    throw new InvalidDataException("NHWC depthwise scalar check failed");
            }
        CheckBudget(budget);
        return new { kernel = "nhwc_depthwise_7x7", channels, height = h, width = w, calls_per_sample = 1, samples_ms_per_call = samples, median_ms = samples.Order().ElementAt(1), finite_output = output.All(float.IsFinite) };
    }
}
