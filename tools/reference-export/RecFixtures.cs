using System.Security.Cryptography;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Independent native REC resize/normalization boundary goldens.</summary>
internal static class RecFixtures
{
    public static int ExportOperators(string[] args)
    {
        if (args.Length != 2 || args[0] != "--out") throw new ArgumentException("Use --export-rec-ops --out <new-directory>.");
        if (Environment.GetEnvironmentVariable("DOTNET_EnableHWIntrinsic") != "0" || Environment.GetEnvironmentVariable("PPOCR_NHWC") != "0" || Environment.GetEnvironmentVariable("SIMD_OCR_NOFUSE") != "1")
            throw new InvalidOperationException("Operator goldens require scalar canonical process controls before startup.");
        BundleWriter writer = new(args[1]);
        List<CaseEntry> cases = [];
        void Capture(string id, int[] inputShape, int[] outputShape, byte[] graph, bool exact)
        {
            byte[] onnx = Join(Number(1, 8), Bytes(8, Number(2, 13)), Bytes(7, graph));
            using Model model = Model.Load(onnx);
            using InferenceSession session = new(model, inputShape);
            int count = inputShape.Aggregate(1, (product, value) => checked(product * value));
            float[] input = Enumerable.Range(0, count).Select(index => ((index * 37 + 11) % 101 - 50) / 13f).ToArray();
            string directory = "cases/" + id;
            List<Artifact> artifacts = [
                writer.WriteJson("fixture", directory + "/fixture.json", new { inputShape, outputShape, exact, operators = model.Nodes.Select(node => node.OpType).ToArray() }),
                writer.WriteBytes("model", directory + "/model.onnx", onnx, [onnx.Length], "onnx"),
                writer.WriteFloats("input", directory + "/input.f32", input, inputShape, "nchw")
            ];
            ReadOnlySpan<float> output = session.RunWithTrace(input, (index, name, op, shape, nhwc, values) =>
            {
                artifacts.Add(writer.WriteFloats($"node.{index:D4}", $"{directory}/node-{index:D4}.f32", values, shape.ToArray(), shape.Length == 4 ? "nchw" : "row_major"));
            });
            if (!session.OutputShape.Dimensions.SequenceEqual(outputShape)) throw new InvalidDataException("Native operator fixture shape differs: " + id);
            artifacts.Add(writer.WriteFloats("output", directory + "/output.f32", output, outputShape, outputShape.Length == 4 ? "nchw" : "row_major"));
            cases.Add(new(id, "recognition_operator", artifacts));
        }
        byte[] Graph(int[] input, int[] output, byte[] node, params byte[][] tensors) =>
            Join(new[] { node }.Concat(tensors).Concat(new[] { Bytes(11, Info("x", input)), Bytes(12, Info("y", output)) }).ToArray());
        Capture("batch-normalization", [2, 2, 2, 3], [2, 2, 2, 3],
            Graph([2, 2, 2, 3], [2, 2, 2, 3], Node("bn", "BatchNormalization", ["x", "scale", "bias", "mean", "variance"], "y", FloatAttribute("epsilon", 1e-5f)),
                Tensor("scale", [2], [1.25f, -.75f]), Tensor("bias", [2], [.25f, -.5f]), Tensor("mean", [2], [.5f, -1f]), Tensor("variance", [2], [.25f, 2f])), false);
        Capture("average-pool", [2, 2, 7, 5], [2, 2, 2, 2],
            Graph([2, 2, 7, 5], [2, 2, 2, 2], Node("pool", "AveragePool", ["x"], "y", IntsAttribute("kernel_shape", [3, 2]), IntsAttribute("strides", [3, 2]))), false);
        Capture("rank-three-matmul", [2, 3, 4], [2, 3, 5],
            Graph([2, 3, 4], [2, 3, 5], Node("projection", "MatMul", ["x", "weights"], "y"), Tensor("weights", [4, 5], Enumerable.Range(0, 20).Select(index => (index % 7 - 3) / 8f).ToArray())), true);
        Capture("rank-three-softmax", [2, 3, 5], [2, 3, 5],
            Graph([2, 3, 5], [2, 3, 5], Node("probability", "Softmax", ["x"], "y", IntAttribute("axis", -1))), false);
        Capture("negative-axis-reduce-mean", [2, 2, 3, 4], [2, 2, 1, 1],
            Graph([2, 2, 3, 4], [2, 2, 1, 1], Node("reduce", "ReduceMean", ["x"], "y", IntsAttribute("axes", [-2, -1]), IntAttribute("keepdims", 1))), false);
        Capture("negative-axis-squeeze", [1, 2, 1, 3], [2, 3],
            Graph([1, 2, 1, 3], [2, 3], Node("squeeze", "Squeeze", ["x", "axes"], "y"), IntegerTensor("axes", [-4, -2])), true);
        // The reference inserts positive axes sequentially. Its negative-axis
        // clamping is outside the shipped tiny model's operator contract.
        Capture("positive-axis-unsqueeze", [2, 3], [1, 2, 1, 3],
            Graph([2, 3], [1, 2, 1, 3], Node("unsqueeze", "Unsqueeze", ["x", "axes"], "y"), IntegerTensor("axes", [0, 2])), true);
        Capture("rank-three-transpose", [2, 3, 4], [2, 4, 3],
            Graph([2, 3, 4], [2, 4, 3], Node("transpose", "Transpose", ["x"], "y", IntsAttribute("perm", [0, 2, 1]))), true);
        byte[] source = File.ReadAllBytes(Path.Combine(Directory.GetCurrentDirectory(), "tools", "reference-export", "RecFixtures.cs"));
        Artifact sourceArtifact = writer.WriteBytes("fixture_source", "source/RecFixtures.cs", source, [source.Length], "source_bytes");
        writer.WriteManifest(new {
            schemaVersion = 1, contract = new { profile = "canonical", backend = "cpu", logicalInputLayout = "nchw", fixtureKind = "recognition-operators-v1", arithmetic = "original C# InferenceSession, scalar unfused; ONNX fixtures generated independently in C#" }, models = Array.Empty<ModelAsset>(), cases,
            auxiliaryFiles = new[] { new { path = sourceArtifact.Path, sha256 = sourceArtifact.Sha256, bytes = sourceArtifact.Bytes, role = "source" } },
            provenance = new { implementation = "original Sdcb.SimdPaddleOCR operator kernels", framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription, assemblySha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(Model).Assembly.Location))).ToLowerInvariant() }
        });
        Console.WriteLine($"Recognition operator goldens written: {cases.Count} cases -> {writer.RootPath}");
        return 0;
    }

    /// <summary>Additive all-model scalar fixtures; never rewrites the v1 captures.</summary>
    public static int ExportScalarOperators(string[] args)
    {
        if (args.Length != 2 || args[0] != "--out") throw new ArgumentException("Use --export-scalar-rec-ops --out <new-directory>.");
        Step12Tools.RequireCanonical();
        BundleWriter writer = new(args[1]);
        List<CaseEntry> cases = [];
        void Capture(string id, int[] inputShape, int[] outputShape, byte[] graph, bool exact, float[]? supplied = null)
        {
            byte[] onnx = Join(Number(1, 8), Bytes(8, Number(2, 13)), Bytes(7, graph));
            using Model model = Model.Load(onnx);
            using InferenceSession session = new(model, inputShape, 1);
            int count = inputShape.Aggregate(1, (product, value) => checked(product * value));
            float[] input = supplied ?? Enumerable.Range(0, count).Select(index => ((index * 37 + 11) % 101 - 50) / 13f).ToArray();
            if (input.Length != count) throw new InvalidDataException("Operator fixture input length differs: " + id);
            string directory = "cases/" + id;
            List<Artifact> artifacts = [writer.WriteJson("fixture", directory + "/fixture.json", new { inputShape, outputShape, exact, operators = model.Nodes.Select(node => node.OpType).ToArray() }),
                writer.WriteBytes("model", directory + "/model.onnx", onnx, [onnx.Length], "onnx"),
                writer.WriteFloats("input", directory + "/input.f32", input, inputShape, inputShape.Length == 4 ? "nchw" : "row_major")];
            ReadOnlySpan<float> output = session.RunWithTrace(input, (index, name, op, shape, nhwc, values) =>
                artifacts.Add(writer.WriteFloats($"node.{index:D4}", $"{directory}/node-{index:D4}.f32", values, shape.ToArray(), shape.Length == 4 ? "nchw" : "row_major")));
            if (!session.OutputShape.Dimensions.SequenceEqual(outputShape)) throw new InvalidDataException("Native scalar operator fixture shape differs: " + id);
            artifacts.Add(writer.WriteFloats("output", directory + "/output.f32", output, outputShape, outputShape.Length == 4 ? "nchw" : "row_major"));
            cases.Add(new(id, "recognition_operator", artifacts));
        }
        byte[] Graph(int[] input, int[] output, byte[] nodes, params byte[][] tensors) =>
            Join(new[] { nodes }.Concat(tensors).Concat(new[] { Bytes(11, Info("x", input)), Bytes(12, Info("y", output)) }).ToArray());
        Capture("row-scalar-subtraction", [2, 3, 7], [2, 3, 7],
            Graph([2, 3, 7], [2, 3, 7], Node("subtract", "Sub", ["x", "mean"], "y"), Tensor("mean", [2, 3, 1], [.1f, -.2f, 1f, 0f, .5f, -1f])), true);
        Capture("left-row-scalar-division", [2, 3, 7], [2, 3, 7],
            Graph([2, 3, 7], [2, 3, 7], Node("divide", "Div", ["scale", "x"], "y"), Tensor("scale", [2, 3, 1], [.1f, -.2f, 1f, 0f, .5f, -1f])), true,
            Enumerable.Range(0, 42).Select(i => (i + 1) / 9f).ToArray());
        float[] squareInput = [0f, -0f, float.Epsilon, -float.Epsilon, 1e-19f, -1e-19f, 1e-10f, -.3f, .3f, 1f, -1f, 123.45f, -123.45f, 1e10f];
        Capture("scalar-square", [2, 7], [2, 7], Graph([2, 7], [2, 7], Node("square", "Pow", ["x", "exponent"], "y"), Tensor("exponent", [], [2f])), false, squareInput);
        Capture("square-root", [2, 7], [2, 7], Graph([2, 7], [2, 7], Node("root", "Sqrt", ["x"], "y")), false,
            squareInput.Select(MathF.Abs).ToArray());
        Capture("trailing-axis-mean", [2, 3, 7], [2, 3, 1], Graph([2, 3, 7], [2, 3, 1], Node("mean", "ReduceMean", ["x"], "y", IntsAttribute("axes", [-1]), IntAttribute("keepdims", 1))), true);
        Capture("dynamic-rank-four-attention-scores", [2, 3, 5, 7], [2, 3, 5, 5],
            Graph([2, 3, 5, 7], [2, 3, 5, 5], Join(Node("transpose-key", "Transpose", ["x"], "key", IntsAttribute("perm", [0, 1, 3, 2])), Node("scores", "MatMul", ["x", "key"], "y"))), true);
        Capture("dynamic-rank-four-attention-context", [2, 3, 5, 7], [2, 3, 5, 7],
            Graph([2, 3, 5, 7], [2, 3, 5, 7], Join(Node("transpose-key", "Transpose", ["x"], "key", IntsAttribute("perm", [0, 1, 3, 2])),
                Node("scores", "MatMul", ["x", "key"], "scores"), Node("probability", "Softmax", ["scores"], "probability", IntAttribute("axis", 3)),
                Node("context", "MatMul", ["probability", "x"], "y"))), false);
        Capture("broadcasted-rank-four-matmul", [2, 3, 5, 7], [2, 3, 5, 9],
            Graph([2, 3, 5, 7], [2, 3, 5, 9], Node("broadcasted", "MatMul", ["x", "weights"], "y"),
                Tensor("weights", [1, 3, 7, 9], Enumerable.Range(0, 189).Select(i => (i % 11 - 5) / 16f).ToArray())), true);
        for (int part = 0; part < 3; part++)
            Capture("qkv-slice-" + part, [3, 2, 3, 5, 7], [2, 3, 5, 7],
                Graph([3, 2, 3, 5, 7], [2, 3, 5, 7], Join(Node("slice", "Slice", ["x", "start", "end", "axis", "step"], "slice"), Node("squeeze", "Squeeze", ["slice", "axis"], "y")),
                    IntegerTensor("start", [part]), IntegerTensor("end", [part + 1]), IntegerTensor("axis", [0]), IntegerTensor("step", [1])), true);
        Capture("strided-data-slice", [2, 7, 5], [2, 3, 5], Graph([2, 7, 5], [2, 3, 5], Node("slice", "Slice", ["x", "start", "end", "axis", "step"], "y"),
            IntegerTensor("start", [1]), IntegerTensor("end", [6]), IntegerTensor("axis", [1]), IntegerTensor("step", [2])), true);
        Capture("negative-step-data-slice", [2, 7, 5], [2, 4, 5], Graph([2, 7, 5], [2, 4, 5], Node("slice", "Slice", ["x", "start", "end", "axis", "step"], "y"),
            IntegerTensor("start", [6]), IntegerTensor("end", [long.MinValue]), IntegerTensor("axis", [-2]), IntegerTensor("step", [-2])), true);
        byte[] LayerNormGraph() => Graph([2, 3, 7], [2, 3, 7], Join(
            Node("mean", "ReduceMean", ["x"], "mean", IntsAttribute("axes", [-1]), IntAttribute("keepdims", 1)),
            Node("center", "Sub", ["x", "mean"], "center"), Node("square", "Pow", ["center", "exponent"], "square"),
            Node("variance", "ReduceMean", ["square"], "variance", IntsAttribute("axes", [-1]), IntAttribute("keepdims", 1)),
            Node("epsilon", "Add", ["variance", "epsilon"], "stabilized"), Node("root", "Sqrt", ["stabilized"], "root"),
            Node("normalize", "Div", ["center", "root"], "normalized"), Node("scale", "Mul", ["normalized", "gamma"], "scaled"), Node("bias", "Add", ["scaled", "beta"], "y")),
            Tensor("exponent", [], [2f]), Tensor("epsilon", [], [1e-5f]), Tensor("gamma", [7], [1f, .5f, 2f, -1f, .75f, -.25f, 1.25f]), Tensor("beta", [7], [0f, .1f, -.2f, .3f, -.4f, .5f, -.6f]));
        Capture("unfused-layer-normalization", [2, 3, 7], [2, 3, 7], LayerNormGraph(), false);
        Capture("near-constant-layer-normalization", [2, 3, 7], [2, 3, 7], LayerNormGraph(), false,
            Enumerable.Range(0, 42).Select(i => i < 7 ? 1f : i < 14 ? 10000f + (i % 3 - 1) / 1024f : i < 21 ? 1e-6f + (i % 3 - 1) * 1e-9f : (i % 7 - 3) / 13f).ToArray());
        byte[] dynamicGraph = Join(Node("shape", "Shape", ["x"], "shape"), Node("shape-alias", "Identity", ["shape"], "alias"),
            Node("width", "Slice", ["alias", "start", "end", "axis", "step"], "width"), Node("scalar-width", "Squeeze", ["width", "axis"], "scalar"),
            Node("vector-width", "Unsqueeze", ["scalar", "axis"], "vector"), Node("target", "Concat", ["prefix", "vector"], "target", IntAttribute("axis", 0)),
            Node("reshape", "Reshape", ["x", "target"], "y"), IntegerTensor("start", [3]), IntegerTensor("end", [4]), IntegerTensor("axis", [0]), IntegerTensor("step", [1]),
            IntegerTensor("prefix", [1, 3, 48]), Bytes(11, DynamicInfo("x", [1, 3, 48, "width"])), Bytes(12, DynamicInfo("y", [1, 3, 48, "width"])));
        Capture("dynamic-shape-alias-width32", [1, 3, 48, 32], [1, 3, 48, 32], dynamicGraph, true);
        Capture("dynamic-shape-alias-width64", [1, 3, 48, 64], [1, 3, 48, 64], dynamicGraph, true);
        byte[] source = File.ReadAllBytes(Path.Combine(Directory.GetCurrentDirectory(), "tools", "reference-export", "RecFixtures.cs"));
        Artifact sourceArtifact = writer.WriteBytes("fixture_source", "source/RecFixtures.cs", source, [source.Length], "source_bytes");
        writer.WriteManifest(new { schemaVersion = 1,
            contract = new { profile = "canonical", backend = "cpu", logicalInputLayout = "nchw", fixtureKind = "recognition-scalar-operators-v2", arithmetic = "original C# InferenceSession, serial scalar unfused; Pow uses MathF.Pow; each canonical node captured" },
            models = Array.Empty<ModelAsset>(), cases, auxiliaryFiles = new[] { new { path = sourceArtifact.Path, sha256 = sourceArtifact.Sha256, bytes = sourceArtifact.Bytes, role = "source" } },
            provenance = new { implementation = "original Sdcb.SimdPaddleOCR all-model scalar operator kernels", framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
                assemblySha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(Model).Assembly.Location))).ToLowerInvariant() } });
        Console.WriteLine($"All-model scalar operator goldens written: {cases.Count} cases -> {writer.RootPath}");
        return 0;
    }

    private static byte[] Join(params byte[][] chunks) => chunks.SelectMany(chunk => chunk).ToArray();
    private static byte[] Varint(ulong value)
    {
        List<byte> result = [];
        while (value >= 128) { result.Add((byte)((value & 127) | 128)); value >>= 7; }
        result.Add((byte)value); return result.ToArray();
    }
    private static byte[] Number(int field, long value) => Join(Varint((ulong)field << 3), Varint(unchecked((ulong)value)));
    private static byte[] Bytes(int field, byte[] value) => Join(Varint(((ulong)field << 3) | 2), Varint((ulong)value.Length), value);
    private static byte[] Text(int field, string value) => Bytes(field, System.Text.Encoding.UTF8.GetBytes(value));
    private static byte[] Info(string name, int[] shape) => Join(Text(1, name), Bytes(2, Bytes(1, Join(Number(1, 1), Bytes(2, Join(shape.Select(value => Bytes(1, Number(1, value))).ToArray()))))));
    private static byte[] DynamicInfo(string name, object[] shape) => Join(Text(1, name), Bytes(2, Bytes(1, Join(Number(1, 1), Bytes(2, Join(shape.Select(value => Bytes(1, value is int dimension ? Number(1, dimension) : Text(2, (string)value))).ToArray()))))));
    private static byte[] Node(string name, string op, string[] inputs, string output, params byte[][] attributes) => Bytes(1, Join(Join(inputs.Select(value => Text(1, value)).ToArray()), Text(2, output), Text(3, name), Text(4, op), Join(attributes.Select(value => Bytes(5, value)).ToArray())));
    private static byte[] IntAttribute(string name, long value) => Join(Text(1, name), Number(3, value), Number(20, 2));
    private static byte[] IntsAttribute(string name, long[] values) => Join(Text(1, name), Join(values.Select(value => Number(8, value)).ToArray()), Number(20, 7));
    private static byte[] FloatAttribute(string name, float value) => Join(Text(1, name), Varint(2 << 3 | 5), BitConverter.GetBytes(value), Number(20, 1));
    private static byte[] Tensor(string name, int[] shape, float[] values) => Bytes(5, Join(Join(shape.Select(value => Number(1, value)).ToArray()), Number(2, 1), Text(8, name), Bytes(9, Join(values.Select(BitConverter.GetBytes).ToArray()))));
    private static byte[] IntegerTensor(string name, long[] values) => Bytes(5, Join(Number(1, values.Length), Number(2, 7), Text(8, name), Bytes(9, Join(values.Select(BitConverter.GetBytes).ToArray()))));

    public static int Export(string[] args)
    {
        Dictionary<string, string> options = [];
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 == args.Length || args[i] is not ("--out" or "--reference-repo" or "--exporter-source") || !options.TryAdd(args[i], args[i + 1]))
                throw new ArgumentException("Use --export-rec-fixtures --out <new-directory> --reference-repo <C#-checkout> --exporter-source <exporter-directory>.");
        }
        if (!options.TryGetValue("--out", out string? output)) throw new ArgumentException("Missing --out.");
        if (Environment.GetEnvironmentVariable("DOTNET_EnableHWIntrinsic") != "0")
            throw new InvalidOperationException("Recognition goldens require DOTNET_EnableHWIntrinsic=0 before process startup.");
        BundleWriter writer = new(output);
        List<CaseEntry> cases = [];
        foreach ((int width, int height, bool adaptive, int fixedWidth) in new[] {
            (1, 96, true, 320), (1, 1, true, 320),
            (31, 48, true, 320), (32, 48, true, 320), (33, 48, true, 320),
            (63, 48, true, 320), (64, 48, true, 320), (65, 48, true, 320),
            (37, 29, true, 320), (30, 20, true, 320),
            (120, 20, false, 64), (1, 96, false, 320), (37, 29, false, 32), (1, 1, false, 32) })
        {
            long natural = checked(((long)48 * width + height - 1) / height);
            int targetWidth = adaptive ? checked((int)Math.Max(32, ((natural + 31) / 32) * 32)) : fixedWidth;
            foreach (ImagePixelFormat format in Enum.GetValues<ImagePixelFormat>())
            {
                int bytesPerPixel = format is ImagePixelFormat.Bgr24 or ImagePixelFormat.Rgb24 ? 3 : 4;
                int stride = checked(width * bytesPerPixel + 13);
                // Final row deliberately omits trailing padding, matching the
                // public image-view source-size contract.
                byte[] pixels = new byte[checked((height - 1) * stride + width * bytesPerPixel)];
                Array.Fill(pixels, (byte)243);
                for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
                {
                    int p = (y * width + x) * 3, destination = y * stride + x * bytesPerPixel;
                    byte blue = (byte)((p * 73 + 19) % 256), green = (byte)(((p + 1) * 73 + 19) % 256), red = (byte)(((p + 2) * 73 + 19) % 256);
                    bool rgb = format is ImagePixelFormat.Rgb24 or ImagePixelFormat.Rgba32;
                    pixels[destination] = rgb ? red : blue;
                    pixels[destination + 1] = green;
                    pixels[destination + 2] = rgb ? blue : red;
                    if (bytesPerPixel == 4) pixels[destination + 3] = (byte)((x * 17 + y * 31) % 256);
                }
                // A dirty destination establishes that padding is refreshed.
                float[] tensor = Enumerable.Repeat(123f, checked(3 * 48 * targetWidth)).ToArray();
                int actualWidth = PPOCRPreprocess.Rec(pixels, width, height, stride, targetWidth, tensor, format);
                string id = $"rec-{width}x{height}-{(adaptive ? "adaptive" : "fixed" + fixedWidth)}-{format.ToString().ToLowerInvariant()}";
                string directory = $"cases/{id}";
                cases.Add(new(id, "rec_preprocess", [
                    writer.WriteJson("fixture", $"{directory}/fixture.json", new { width, height, stride, pixelFormat = format.ToString().ToLowerInvariant(), adaptiveWidth = adaptive, fixedWidth, targetWidth, actualWidth }),
                    writer.WriteBytes("pixels", $"{directory}/pixels.bin", pixels, [pixels.Length], "row_major"),
                    writer.WriteFloats("rec.input", $"{directory}/rec.f32", tensor, [1, 3, 48, targetWidth], "nchw")
                ]));
            }
        }
        List<object> auxiliary = [];
        List<object> sources = [];
        void Snapshot(string original, string relative)
        {
            byte[] bytes = File.ReadAllBytes(original);
            Artifact artifact = writer.WriteBytes("source." + sources.Count, relative, bytes, [bytes.Length], "source_bytes");
            auxiliary.Add(new { path = artifact.Path, sha256 = artifact.Sha256, bytes = artifact.Bytes, role = "source" });
            sources.Add(new { path = relative, sha256 = artifact.Sha256, bytes = artifact.Bytes });
        }
        if (options.TryGetValue("--reference-repo", out string? reference))
            Snapshot(Path.Combine(reference, "src", "Sdcb.SimdPaddleOCR", "PPOCRPreprocess.cs"), "source/PPOCRPreprocess.cs");
        if (options.TryGetValue("--exporter-source", out string? exporter))
            Snapshot(Path.Combine(exporter, "RecFixtures.cs"), "source/RecFixtures.cs");
        string assembly = typeof(PPOCRPreprocess).Assembly.Location;
        writer.WriteManifest(new {
            schemaVersion = 1,
            contract = new { profile = "canonical", backend = "cpu", logicalInputLayout = "nchw", fixtureKind = "recognition-v1", decoder = "none; deterministic generated pixels", arithmetic = "original C# REC preprocessing with hardware intrinsics disabled", finalRow = "no trailing row padding" },
            models = Array.Empty<ModelAsset>(), cases, auxiliaryFiles = auxiliary,
            provenance = new { implementation = "original Sdcb.SimdPaddleOCR REC preprocessing", framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription, os = System.Runtime.InteropServices.RuntimeInformation.OSDescription, assemblySha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(assembly))).ToLowerInvariant(), sources }
        });
        Console.WriteLine($"Recognition goldens written: {cases.Count} cases -> {writer.RootPath}");
        return 0;
    }
}
