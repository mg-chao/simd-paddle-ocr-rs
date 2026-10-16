using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

// The assembly name deliberately matches the reference's existing friend
// assembly. No reference source is copied, patched or generated into this app.
internal static class Program
{
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
    };
    private static readonly string[] Models = ["cls", "det", "small_det", "medium_det", "rec", "small_rec", "medium_rec"];
    private static string _output = "";
    private static readonly List<object> Cases = [];
    private static int _iterations;
    private static int _threads;
    private static int _batchLines;
    private static bool _referenceMode;
    private static bool _correctnessOnly;

    private static int Main(string[] args)
    {
        try { Run(args); return 0; }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    private static void Run(string[] args)
    {
        if (args.Contains("--acceptance")) { AcceptanceRunner.Run(args); return; }
        Dictionary<string, string> options = [];
        HashSet<string> flags = [];
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] is "--quick" or "--generate-corpus" or "--imports-only" or "--character-oracle" or "--batch-only" or "--correctness-only") flags.Add(args[i]);
            else if (args[i] is "--corpus" or "--sample" or "--models-dir" or "--output" or "--mode" or "--iterations" or "--threads" or "--ocr-batch-size"
                     && i + 1 < args.Length) options.Add(args[i++], args[i]);
            else throw new ArgumentException($"Unknown or incomplete argument: {args[i]}");
        }
        string Required(string name) => options.TryGetValue(name, out string? value) ? Path.GetFullPath(value) : throw new ArgumentException($"Missing {name}");
        if (flags.Contains("--character-oracle")) { CharacterOracle.Write(Required("--output")); return; }
        if (flags.Contains("--imports-only"))
        {
            foreach (string name in Models) ImportCapture.Write(Path.Combine(Required("--models-dir"), name + ".onnx"), Required("--output"), name);
            return;
        }
        string corpus = Required("--corpus");
        if (flags.Contains("--generate-corpus")) { GenerateCorpus(Required("--sample"), corpus); return; }
        string models = Required("--models-dir");
        _output = Required("--output");
        string mode = options.GetValueOrDefault("--mode", "auto");
        if (mode is not ("reference" or "auto")) throw new ArgumentException("--mode must be reference or auto");
        _referenceMode = mode == "reference";
        _correctnessOnly = flags.Contains("--correctness-only");
        _iterations = int.Parse(options.GetValueOrDefault("--iterations", "5"));
        _threads = int.Parse(options.GetValueOrDefault("--threads", "1"));
        _batchLines = int.Parse(options.GetValueOrDefault("--ocr-batch-size", "1"));
        if (_iterations < 1) throw new ArgumentException("--iterations must be positive");
        if (_threads is < 1 or > 16) throw new ArgumentException("--threads must be within 1..=16");
        if (_batchLines < 1) throw new ArgumentException("--ocr-batch-size must be positive");
        Directory.CreateDirectory(_output);
        bool quick = flags.Contains("--quick");
        var images = LoadCorpus(corpus).Where(image => !quick || image.Quick).ToArray();
        if (images.Length == 0) throw new InvalidDataException("Corpus is empty");
        if (flags.Contains("--batch-only")) CaptureBatch(models, images);
        else
        {
            foreach (string name in Models) CaptureGraph(models, name, mode == "reference");
            foreach (RawImage image in images) CapturePreprocess(image);
            CapturePublic(models, images);
        }
        WriteJson(Path.Combine(_output, "capture.json"), new
        {
            SchemaVersion = 1,
            Engine = "csharp",
            Mode = mode,
            Metadata = new
            {
                Framework = RuntimeInformation.FrameworkDescription,
                Architecture = RuntimeInformation.ProcessArchitecture.ToString(),
                ProcessorCount = Environment.ProcessorCount,
                Threads = _threads,
                OcrBatchSize = _batchLines,
                CorrectnessOnly = _correctnessOnly,
                CorpusSha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(corpus, "corpus.json")))),
                HardwareIntrinsics = Environment.GetEnvironmentVariable("DOTNET_EnableHWIntrinsic"),
                Nhwc = Environment.GetEnvironmentVariable("PPOCR_NHWC"),
                NoFuse = Environment.GetEnvironmentVariable("SIMD_OCR_NOFUSE"),
                PeakWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64,
                ImagesDecodedInsideTiming = false,
            },
            Cases,
        });
        Console.WriteLine($"C# {mode}: {Cases.Count} cases -> {_output}");
    }

    private static void CaptureBatch(string root, RawImage[] images)
    {
        string[] ids = ["sample", "tiny", "blank", "sample"];
        RawImage[] selected = ids.Select(id => images.Single(image => image.Id == id)).ToArray();
        int[] widths = selected.Select(image => image.Width).ToArray(), heights = selected.Select(image => image.Height).ToArray();
        int[] lengths = selected.Select(image => image.Bytes.Length).ToArray(), offsets = new int[selected.Length];
        for (int i = 1; i < offsets.Length; i++) offsets[i] = offsets[i - 1] + lengths[i - 1];
        byte[] buffer = selected.SelectMany(image => image.Bytes).ToArray();
        foreach ((string name, string keys) in new[] { ("rec", "ppocr_keys.txt"), ("small_rec", "rec_keys.txt"), ("medium_rec", "rec_keys.txt") })
        {
            using Model model = LoadModel(root, name);
            using var recognizer = new PaddleOcrRecognizer(model, File.ReadAllBytes(Path.Combine(root, keys)), new() { Backend = OcrBackend.Cpu }, ownsModel: false, intraOpThreads: _threads);
            PaddleOcrRecognitionResult[] standalone = selected.Select(image => recognizer.Recognize(image.Bytes, image.Width, image.Height, true)).ToArray();
            int target = recognizer.SelectWidthForCrop(widths[0], heights[0]);
            if (selected.Any(image => recognizer.SelectWidthForCrop(image.Width, image.Height) != target)) throw new InvalidDataException("Batch images must share the selected target width");
            PaddleOcrRecognitionResult[] results = new PaddleOcrRecognitionResult[selected.Length];
            recognizer.RecognizeBatch(buffer, offsets, lengths, widths, heights, [0, 1, 2, 3], target, results, _threads, true);
            Cases.Add(new { Id = $"batch/{name}", ImageIds = ids, Results = results, Standalone = standalone });
            Console.WriteLine($"batch {name}: four same-width inputs");
        }
    }

    private static void CaptureGraph(string root, string name, bool trace)
    {
        var loadTimer = Stopwatch.StartNew();
        using Model model = LoadModel(root, name);
        var compiled = new CompiledModel(model, _threads);
        double loadNs = loadTimer.Elapsed.TotalNanoseconds;
        int[] shape = name == "cls" ? [1, 3, 80, 160] : name.EndsWith("det") ? [1, 3, 32, 32] : [1, 3, 48, 32];
        using var session = compiled.CreateRequest();
        for (int index = 0; index < 3; index++)
        {
            int[] actual = (int[])shape.Clone();
            if (index == 1 && name != "cls") actual[3] = 96;
            var reshapeTimer = Stopwatch.StartNew();
            session.Reshape(actual);
            double reshapeNs = reshapeTimer.Elapsed.TotalNanoseconds;
            float[] input = Enumerable.Range(0, actual.Aggregate(1, (a, b) => a * b)).Select(i => ((i % 97) - 48f) / 64f).ToArray();
            int[] outputShape = session.OutputShape.Dimensions;
            float[] output = new float[outputShape.Aggregate(1, (a, b) => a * b)];
            object benchmark = Measure(() => { session.Run(input, output); return 0; });
            string id = $"graph/{name}/{index}";
            List<object> nodes = [];
            if (trace && index == 0)
            {
                // This hook performs one execution, copying each result before
                // the next node can reuse its storage. It is validated against
                // the pinned source revision by cpu_parity.py before launch.
                var probe = new SessionProbe(session);
                input.CopyTo(session.InputData);
                var shapes = session.ResolvedNodeShapes();
                for (int ni = 0; ni < model.Nodes.Length; ni++)
                {
                    if (probe.Step(ni) != 0) throw new InvalidOperationException("Logical capture requires SIMD_OCR_NOFUSE=1");
                    NodeRecord node = model.Nodes[ni];
                    float[] values = probe.Copy(node.Outputs[0]);
                    nodes.Add(new { Index = ni, Name = node.Name, OpType = node.OpType,
                        Tensor = Tensor($"{id}/node-{ni:D4}.f32", shapes[ni].Shape, values) });
                }
            }
            Cases.Add(new { Id = id, InputShape = actual, Tensor = Tensor($"{id}/output.f32", outputShape, output), Nodes = nodes,
                Benchmark = benchmark, LoadCompileNs = loadNs, ReshapeNs = reshapeNs, RetainedActivationBytes = new SessionProbe(session).WorkspaceBytes });
            Console.WriteLine($"graph {name}/{index} [{string.Join(',', actual)}] -> [{string.Join(',', outputShape)}]");
        }
    }

    private static void CapturePreprocess(RawImage image)
    {
        var size = PPOCRPreprocess.ComputeDetSize(image.Width, image.Height, 320);
        float[] det = new float[3 * size.Width * size.Height];
        PPOCRPreprocess.Det(image.Bytes, image.Width, image.Height, image.Width * 3, size.Width, size.Height, det);
        Cases.Add(new { Id = $"preprocess/det/{image.Id}", Tensor = Tensor($"preprocess/det/{image.Id}.f32", [1, 3, size.Height, size.Width], det) });
        float[] cls = new float[3 * 80 * 160];
        PPOCRPreprocess.Cls(image.Bytes, image.Width, image.Height, image.Width * 3, cls);
        Cases.Add(new { Id = $"preprocess/cls/{image.Id}", Tensor = Tensor($"preprocess/cls/{image.Id}.f32", [1, 3, 80, 160], cls) });
        float[] rec = new float[3 * 48 * 320];
        PPOCRPreprocess.Rec(image.Bytes, image.Width, image.Height, image.Width * 3, 320, rec);
        Cases.Add(new { Id = $"preprocess/rec/{image.Id}", Tensor = Tensor($"preprocess/rec/{image.Id}.f32", [1, 3, 48, 320], rec) });
        PaddleOcrDetectionBox box = new(0, 0, image.Width - 1, 0, image.Width - 1, image.Height - 1, 0, image.Height - 1, 1);
        if (image.Width >= 9 && image.Height >= 9)
            box = new(1.25f, 2f, image.Width - 2f, 1.5f, image.Width - 1f, image.Height - 2f, .5f, image.Height - 1f, 1);
        byte[] crop = PPOCRCrop.Extract(image.Bytes, image.Width, image.Height, image.Width * 3, box, out int width, out int height);
        Cases.Add(new { Id = $"crop/{image.Id}", Tensor = Tensor($"crop/{image.Id}.f32", [height, width, 3], crop.Select(x => (float)x).ToArray()) });
    }

    private static void CapturePublic(string root, RawImage[] images)
    {
        using Model clsModel = LoadModel(root, "cls");
        using var classifier = new PaddleOcrClassifier(clsModel, new() { Backend = OcrBackend.Cpu });
        foreach (RawImage image in images.Where(image => image.PublicCase))
        {
            PaddleOcrClassificationResult result = default;
            object benchmark = Measure(() => result = classifier.Classify(image.Bytes, image.Width, image.Height));
            Cases.Add(new { Id = $"classify/{image.Id}", Result = new { result.Label, result.Score, result.ResizedWidth }, Benchmark = benchmark });
        }
        foreach ((string detName, string recName, string dictionary) in new[] { ("det", "rec", "ppocr_keys.txt"), ("small_det", "small_rec", "rec_keys.txt"), ("medium_det", "medium_rec", "rec_keys.txt") })
        {
            using Model detModel = LoadModel(root, detName);
            using Model recModel = LoadModel(root, recName);
            byte[] keys = File.ReadAllBytes(Path.Combine(root, dictionary));
            using var recognizer = new PaddleOcrRecognizer(recModel, keys, new() { Backend = OcrBackend.Cpu }, ownsModel: false, intraOpThreads: _threads);
            using var detector = new PaddleOcrDetector(detModel, new() { Backend = OcrBackend.Cpu, LimitSideLength = 320 }, _threads);
            using var all = new PaddleOcrAll(detModel, clsModel, recModel, keys, new()
            {
                LineWorkerCount = Math.Min(_threads, 4), DetIntraOpThreads = _threads, PreprocessWorkerCount = _threads, RecBatchLines = _batchLines,
                Detector = new() { Backend = OcrBackend.Cpu, LimitSideLength = 320 },
                Classifier = new() { Backend = OcrBackend.Cpu }, Recognizer = new() { Backend = OcrBackend.Cpu },
            });
            foreach (RawImage image in images.Where(image => image.PublicCase))
            {
                PaddleOcrRecognitionResult result = default;
                object benchmark = Measure(() => result = recognizer.Recognize(image.Bytes, image.Width, image.Height, true));
                Cases.Add(new { Id = $"recognize/{recName}/{image.Id}", Result = new
                {
                    result.Text, result.Score, result.EmittedCount, result.ResizedWidth, result.TensorWidth, result.TimeSteps,
                    CtcSpans = result.CtcSpans?.Select(span => new { span.Text, span.Score, span.StartColumn, span.EndColumn }).ToArray(),
                }, Benchmark = benchmark });
                PaddleOcrDetectionResult? detection = null;
                benchmark = Measure(() => detection = detector.Detect(image.Bytes, image.Width, image.Height));
                Cases.Add(new { Id = $"detect/{detName}/{image.Id}", Result = new
                {
                    Boxes = detection!.Boxes.Select(Box).ToArray(), detection.ResizedWidth, detection.ResizedHeight, detection.WidthRatio, detection.HeightRatio,
                }, Benchmark = benchmark });
                if (!image.E2eCase) continue;
                PaddleOcrResult? ocr = null;
                benchmark = Measure(() => ocr = all.Run(image.Bytes, image.Width, image.Height, true));
                Cases.Add(new { Id = $"ocr/{recName}/{image.Id}", Result = new
                {
                    ocr!.Text, ocr.DetectedCount, ocr.DetectorResizedWidth, ocr.DetectorResizedHeight,
                    Lines = ocr.Lines.Select(line => new
                    {
                        Detection = Box(line.Box), line.Text, line.RecognitionScore, line.ClassificationScore, line.ClassificationLabel,
                        line.AppliedRotationDegrees, line.EmittedCount, line.RecognitionContentWidth, line.RecognitionTensorWidth, line.RecognitionTimeSteps,
                        CtcSpans = line.CtcSpans?.Select(span => new { span.Text, span.Score, span.StartColumn, span.EndColumn }).ToArray(),
                    }).ToArray(),
                }, Benchmark = benchmark });
            }
            Console.WriteLine($"public {recName}: {images.Count(image => image.PublicCase)} images");
        }
    }

    private static object Box(PaddleOcrDetectionBox box) => new
    {
        Points = new[] { new[] { box.X1, box.Y1 }, new[] { box.X2, box.Y2 }, new[] { box.X3, box.Y3 }, new[] { box.X4, box.Y4 } }, box.Score,
    };

    private static object Measure<T>(Func<T> operation)
    {
        if (_correctnessOnly)
        {
            operation();
            return new { Measured = false, Reason = "correctness-only" };
        }
        var timer = Stopwatch.StartNew(); operation(); double cold = timer.Elapsed.TotalNanoseconds;
        int warmups = _referenceMode ? 0 : 2;
        for (int i = 0; i < warmups; i++) operation();
        double[] warm = new double[_iterations]; long[] allocations = new long[_iterations];
        for (int i = 0; i < _iterations; i++)
        {
            long before = GC.GetTotalAllocatedBytes(false);
            timer.Restart(); operation(); warm[i] = timer.Elapsed.TotalNanoseconds;
            allocations[i] = GC.GetTotalAllocatedBytes(false) - before;
        }
        return new { ColdNs = cold, WarmupRuns = warmups, WarmNs = warm, AllocatedBytes = allocations };
    }

    private static object Tensor(string relative, int[] shape, float[] values)
    {
        if (values.Any(value => !float.IsFinite(value))) throw new InvalidDataException($"Nonfinite tensor: {relative}");
        string path = Path.Combine(_output, relative.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        byte[] bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        if (!BitConverter.IsLittleEndian) for (int i = 0; i < bytes.Length; i += 4) Array.Reverse(bytes, i, 4);
        File.WriteAllBytes(path, bytes);
        return new { File = relative, Shape = shape, Sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)) };
    }

    private static void WriteJson(string path, object value) => File.WriteAllText(path, JsonSerializer.Serialize(value, Json));

    private static Model LoadModel(string root, string name)
    {
        using Stream stream = File.OpenRead(Path.Combine(root, name + ".onnx"));
        return Model.Load(stream);
    }

    private sealed record RawImage(string Id, int Width, int Height, byte[] Bytes, bool Quick, bool PublicCase, bool E2eCase);

    private static RawImage[] LoadCorpus(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(path, "corpus.json")));
        if (document.RootElement.GetProperty("schema_version").GetInt32() != 1) throw new InvalidDataException("Unknown corpus schema");
        return document.RootElement.GetProperty("images").EnumerateArray().Select(item =>
        {
            string id = item.GetProperty("id").GetString()!;
            byte[] bytes = File.ReadAllBytes(Path.Combine(path, item.GetProperty("file").GetString()!));
            if (Convert.ToHexStringLower(SHA256.HashData(bytes)) != item.GetProperty("sha256").GetString()) throw new InvalidDataException($"Corpus hash mismatch: {id}");
            int width = item.GetProperty("width").GetInt32(), height = item.GetProperty("height").GetInt32();
            if (bytes.Length != checked(width * height * 3)) throw new InvalidDataException($"Corpus size mismatch: {id}");
            return new RawImage(id, width, height, bytes, item.GetProperty("quick").GetBoolean(), item.GetProperty("public_case").GetBoolean(), item.GetProperty("e2e_case").GetBoolean());
        }).ToArray();
    }

    private static void GenerateCorpus(string sample, string directory)
    {
        Directory.CreateDirectory(directory);
        List<object> images = [];
        void Add(string id, int width, int height, byte[] bytes, bool quick, bool publicCase = true, bool e2eCase = true)
        {
            string file = id + ".bgr";
            File.WriteAllBytes(Path.Combine(directory, file), bytes);
            images.Add(new { Id = id, Width = width, Height = height, File = file,
                Sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)), Quick = quick, PublicCase = publicCase, E2eCase = e2eCase });
        }
        // Frozen raw pixels make generation independent of codecs and fonts.
        // sample.json records the original JPEG and one-time decoding provenance.
        using var source = JsonDocument.Parse(File.ReadAllText(Path.Combine(sample, "sample.json")));
        int w = source.RootElement.GetProperty("width").GetInt32(), h = source.RootElement.GetProperty("height").GetInt32();
        byte[] pixels = File.ReadAllBytes(Path.Combine(sample, "sample.bgr"));
        if (pixels.Length != w * h * 3 || Convert.ToHexStringLower(SHA256.HashData(pixels)) != source.RootElement.GetProperty("sha256").GetString())
            throw new InvalidDataException("Frozen source corpus checksum/size mismatch");
        Add("sample", w, h, pixels, true);
        byte[] rotated = new byte[pixels.Length], skew = Enumerable.Repeat((byte)255, pixels.Length).ToArray();
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
        {
            pixels.AsSpan((y * w + x) * 3, 3).CopyTo(rotated.AsSpan(((h - 1 - y) * w + w - 1 - x) * 3, 3));
            int sx = x - (y - h / 2) / 8;
            if ((uint)sx < (uint)w) pixels.AsSpan((y * w + sx) * 3, 3).CopyTo(skew.AsSpan((y * w + x) * 3, 3));
        }
        Add("sample-180", w, h, rotated, false);
        Add("sample-skew", w, h, skew, false);
        Add("text-line", 320, 48, TextImage(320, 48, 1), true);
        Add("text-multiline", 320, 144, TextImage(320, 144, 3), false);
        byte[] horizontal = TextImage(320, 48, 1), vertical = new byte[horizontal.Length];
        for (int y = 0; y < 48; y++) for (int x = 0; x < 320; x++)
            horizontal.AsSpan((y * 320 + x) * 3, 3).CopyTo(vertical.AsSpan((x * 48 + 47 - y) * 3, 3));
        Add("text-vertical", 48, 320, vertical, false);
        Add("text-long", 1024, 48, TextImage(1024, 48, 1), false, e2eCase: false);
        Add("blank", 32, 32, Enumerable.Repeat((byte)255, 32 * 32 * 3).ToArray(), true);
        Add("tiny", 7, 9, Enumerable.Range(0, 7 * 9 * 3).Select(i => (byte)(i * 17 % 256)).ToArray(), true, publicCase: false, e2eCase: false);
        Add("narrow", 2, 128, Enumerable.Range(0, 2 * 128 * 3).Select(i => (byte)(i * 13 % 256)).ToArray(), false, publicCase: false, e2eCase: false);
        uint state = 0xC0FFEE;
        byte[] noise = new byte[64 * 64 * 3];
        for (int i = 0; i < noise.Length; i++) { state = unchecked(state * 1664525 + 1013904223); noise[i] = (byte)(state >> 24); }
        Add("noise", 64, 64, noise, false, e2eCase: false);
        WriteJson(Path.Combine(directory, "corpus.json"), new { SchemaVersion = 1, PixelFormat = "bgr24", GeneratorVersion = 1,
            SourceSampleSha256 = source.RootElement.GetProperty("source_jpeg_sha256").GetString(), Images = images });
        Console.WriteLine($"Generated {images.Count} deterministic raw images at {directory}");
    }

    private static byte[] TextImage(int width, int height, int rows)
    {
        // A tiny explicitly encoded bitmap alphabet avoids host fonts entirely.
        Dictionary<char, string[]> font = new()
        {
            ['R'] = ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
            ['U'] = ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
            ['S'] = ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
            ['T'] = ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
            ['I'] = ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
            ['M'] = ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
            ['D'] = ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
            ['1'] = ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
            ['2'] = ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
            ['3'] = ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        };
        byte[] output = Enumerable.Repeat((byte)255, width * height * 3).ToArray();
        const string text = "RUST SIMD 123 ";
        for (int row = 0; row < rows; row++) for (int character = 0; character < (width - 12) / 18; character++)
        {
            char c = text[character % text.Length]; if (!font.TryGetValue(c, out string[]? glyph)) continue;
            for (int y = 0; y < 21; y++) for (int x = 0; x < 15; x++) if (glyph[y / 3][x / 3] == '1')
            {
                int px = 8 + character * 18 + x, py = 10 + row * 42 + y;
                if (py < height) output.AsSpan((py * width + px) * 3, 3).Clear();
            }
        }
        return output;
    }

    private sealed class SessionProbe
    {
        private static readonly BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
        private static readonly FieldInfo Tensors = typeof(InferenceSession).GetField("_tensors", Private) ?? throw new MissingFieldException("Pinned _tensors hook changed");
        private static readonly FieldInfo Workspace = typeof(InferenceSession).GetField("_workspace", Private) ?? throw new MissingFieldException("Pinned _workspace hook changed");
        private static readonly MethodInfo Execute = typeof(InferenceSession).GetMethod("ExecuteStep", Private) ?? throw new MissingMethodException("Pinned ExecuteStep hook changed");
        private readonly InferenceSession _session;
        private readonly Array _tensors;
        private readonly Func<int, int> _step;
        internal SessionProbe(InferenceSession session)
        {
            _session = session; _tensors = (Array)Tensors.GetValue(session)!;
            _step = Execute.CreateDelegate<Func<int, int>>(session);
        }
        internal long WorkspaceBytes => ((NativeWorkspace?)Workspace.GetValue(_session))?.Length * 4L ?? 0;
        internal int Step(int index) => _step(index);
        internal unsafe float[] Copy(uint tensor)
        {
            object value = _tensors.GetValue(checked((int)tensor))!;
            Type type = value.GetType();
            int offset = (int)type.GetProperty("Offset")!.GetValue(value)!;
            int length = (int)type.GetProperty("Length")!.GetValue(value)!;
            if (type.GetField("_native", Private)!.GetValue(value) is NativeWorkspace native)
                return new ReadOnlySpan<float>(native.Pointer + offset, length).ToArray();
            if (type.GetField("_buffer", Private)!.GetValue(value) is float[] buffer)
                return buffer.AsSpan(offset, length).ToArray();
            throw new InvalidOperationException("Node output has no materialized activation buffer");
        }
    }
}
