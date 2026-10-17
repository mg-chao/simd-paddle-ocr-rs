using System.Text.Json;
using System.Text.Json.Serialization;
using Sdcb.SimdPaddleOCR;

// One model per process, one JSON request/result per line. The parent enforces
// per-case deadlines and supplies already hash-verified, decoded BGR24 bytes.
if (args.Length != 3) throw new ArgumentException("usage: Reference manifest models-directory model-id");
if (Environment.ProcessorCount != 1) throw new InvalidOperationException("DOTNET_PROCESSOR_COUNT must be 1");
using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(args[0]));
JsonElement settings = manifest.RootElement.GetProperty("settings");
JsonElement model = manifest.RootElement.GetProperty("models").EnumerateArray().Single(m => m.GetProperty("id").GetString() == args[2]);
string ModelPath(string field) => Path.Combine(args[1], model.GetProperty(field).GetString()!);
using PaddleOcrAll ocr = PaddleOcrAll.Load(ModelPath("det"), ModelPath("cls"), ModelPath("rec"), ModelPath("dictionary"), new PaddleOcrOptions
{
    UseDirectionClassification = settings.GetProperty("direction_classification").GetBoolean(),
    ClassifierThreshold = settings.GetProperty("classifier_threshold").GetSingle(),
    LineWorkerCount = 1,
    PreprocessWorkerCount = 1,
    DetIntraOpThreads = 1,
    RecBatchLines = settings.GetProperty("recognition_batch_size").GetInt32(),
    Detector = new PaddleOcrDetectorOptions
    {
        Backend = OcrBackend.Cpu,
        LimitSideLength = settings.GetProperty("detector_limit").GetInt32(),
        MaxCandidates = settings.GetProperty("max_candidates").GetInt32(),
        BitmapThreshold = settings.GetProperty("bitmap_threshold").GetSingle(),
        BoxThreshold = model.GetProperty("box_threshold").GetSingle(),
        UnclipRatio = settings.GetProperty("unclip_ratio").GetSingle(),
        UseDilation = settings.GetProperty("use_dilation").GetBoolean(),
    },
    Classifier = new PaddleOcrClassifierOptions { Backend = OcrBackend.Cpu },
    Recognizer = new PaddleOcrRecognizerOptions
    {
        Backend = OcrBackend.Cpu,
        AdaptiveWidth = settings.GetProperty("adaptive_width").GetBoolean(),
        TargetWidth = settings.GetProperty("target_width").GetInt32(),
    },
});
Console.Error.WriteLine($".NET={Environment.Version} CPU-count={Environment.ProcessorCount} AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} FMA={System.Runtime.Intrinsics.X86.Fma.IsSupported} AVX512={System.Runtime.Intrinsics.X86.Avx512F.IsSupported}");
JsonSerializerOptions json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow };
while (Console.ReadLine() is string line)
{
    Input request = JsonSerializer.Deserialize<Input>(line, json) ?? throw new InvalidDataException("null request");
    byte[] pixels = File.ReadAllBytes(request.Path);
    if (request.Stride != checked(request.Width * 3) || pixels.Length != checked(request.Stride * request.Height))
        throw new InvalidDataException("BGR24 buffer dimensions mismatch");
    PaddleOcrResult result = ocr.Run(pixels, request.Width, request.Height, request.Stride, ImagePixelFormat.Bgr24);
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        request.Id,
        result.Text,
        result.DetectedCount,
        result.DetectorResizedWidth,
        result.DetectorResizedHeight,
        Lines = result.Lines.Select(l => new
        {
            Points = new[] { new[] { l.Box.X1, l.Box.Y1 }, new[] { l.Box.X2, l.Box.Y2 }, new[] { l.Box.X3, l.Box.Y3 }, new[] { l.Box.X4, l.Box.Y4 } },
            DetectionScore = l.Box.Score,
            l.Text,
            l.RecognitionScore,
            l.ClassificationScore,
            l.ClassificationLabel,
            l.AppliedRotationDegrees,
            l.EmittedCount,
            l.RecognitionContentWidth,
            l.RecognitionTensorWidth,
            l.RecognitionTimeSteps,
        }),
    }, json));
}
record Input(string Id, string Path, int Width, int Height, int Stride);
