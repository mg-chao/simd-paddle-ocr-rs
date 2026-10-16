using System.Text.Json;
using System.Text.Json.Serialization;
using Sdcb.SimdPaddleOCR;

// Inputs are decoded once by Rust and hash-verified against the serial manifest.
if (args.Length != 4) throw new ArgumentException("usage: Qualification manifest model-directory model-id profile");
using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(args[0]));
using JsonDocument profileDocument = JsonDocument.Parse(File.ReadAllBytes(args[3]));
JsonElement settings = manifest.RootElement.GetProperty("settings");
JsonElement profile = profileDocument.RootElement;
JsonElement model = manifest.RootElement.GetProperty("models").EnumerateArray().Single(m => m.GetProperty("id").GetString() == args[2]);
string ModelPath(string field) => Path.Combine(args[1], model.GetProperty(field).GetString()!);
int Int(string field) => profile.GetProperty(field).GetInt32();
bool Bool(string field) => profile.GetProperty(field).GetBoolean();
PaddleOcrDetectorOptions detector = Bool("default_thresholds")
    ? new() { Backend = OcrBackend.Cpu, LimitSideLength = settings.GetProperty("detector_limit").GetInt32(), MaxCandidates = settings.GetProperty("max_candidates").GetInt32(), BitmapThreshold = settings.GetProperty("bitmap_threshold").GetSingle(), UnclipRatio = settings.GetProperty("unclip_ratio").GetSingle(), UseDilation = Bool("dilation"), MaxImagePixels = Int("max_image_pixels") }
    : new() { Backend = OcrBackend.Cpu, LimitSideLength = settings.GetProperty("detector_limit").GetInt32(), MaxCandidates = settings.GetProperty("max_candidates").GetInt32(), BitmapThreshold = settings.GetProperty("bitmap_threshold").GetSingle(), BoxThreshold = model.GetProperty("box_threshold").GetSingle(), UnclipRatio = settings.GetProperty("unclip_ratio").GetSingle(), UseDilation = Bool("dilation"), MaxImagePixels = Int("max_image_pixels") };
using PaddleOcrAll ocr = PaddleOcrAll.Load(ModelPath("det"), ModelPath("cls"), ModelPath("rec"), ModelPath("dictionary"), new PaddleOcrOptions {
    UseDirectionClassification = Bool("classification"),
    ClassifierThreshold = settings.GetProperty("classifier_threshold").GetSingle(),
    LineWorkerCount = Int("line_workers"), PreprocessWorkerCount = Int("preprocess_workers"), DetIntraOpThreads = Int("detector_threads"), RecBatchLines = Int("batch_size"),
    Detector = detector, Classifier = new() { Backend = OcrBackend.Cpu },
    Recognizer = new() { Backend = OcrBackend.Cpu, AdaptiveWidth = Bool("adaptive_width"), TargetWidth = Int("target_width") },
});
Console.Error.WriteLine($".NET={Environment.Version} CPU-count={Environment.ProcessorCount} AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} FMA={System.Runtime.Intrinsics.X86.Fma.IsSupported} AVX512={System.Runtime.Intrinsics.X86.Avx512F.IsSupported}");
JsonSerializerOptions json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow };
while (Console.ReadLine() is string line) {
    Input request = JsonSerializer.Deserialize<Input>(line, json) ?? throw new InvalidDataException("null request");
    object response;
    try {
        byte[] pixels = File.ReadAllBytes(request.Path);
        ImagePixelFormat format = Enum.Parse<ImagePixelFormat>(request.Format, ignoreCase: false);
        PaddleOcrResult result = ocr.Run(pixels, request.Width, request.Height, Bool("capture"), request.Stride, format);
        response = new {
            request.Id,
            Result = new { request.Id, result.Text, result.DetectedCount, result.DetectorResizedWidth, result.DetectorResizedHeight,
                Lines = result.Lines.Select(l => new { Points = Points(l.Box), DetectionScore = l.Box.Score, l.Text, l.RecognitionScore, l.ClassificationScore, l.ClassificationLabel, l.AppliedRotationDegrees, l.EmittedCount, l.RecognitionContentWidth, l.RecognitionTensorWidth, l.RecognitionTimeSteps }) },
            Alignments = result.Lines.Select(l => new {
                Spans = l.CtcSpans,
                CharacterBoxes = Bool("capture") ? l.EstimateCharacterBoxes().Select(b => new { b.Text, b.Score, Points = new[] { new[] { b.X1,b.Y1 }, new[] { b.X2,b.Y2 }, new[] { b.X3,b.Y3 }, new[] { b.X4,b.Y4 } } }).ToArray() : null,
            }), Error = (string?)null,
        };
    } catch (Exception error) {
        response = new { request.Id, Result = (object?)null, Alignments = Array.Empty<object>(), Error = error.Message };
    }
    Console.WriteLine(JsonSerializer.Serialize(response, json));
}
static float[][] Points(PaddleOcrDetectionBox b) => [ [b.X1,b.Y1], [b.X2,b.Y2], [b.X3,b.Y3], [b.X4,b.Y4] ];
record Input(string Id, string Fixture, string Path, int Width, int Height, int Stride, string Format, string? ExpectedError);
