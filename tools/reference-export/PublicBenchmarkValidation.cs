using System.Text;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Untimed ordinary public calls; no tracing or graph hooks can alter the optimized path.</summary>
internal static class PublicBenchmarkValidation
{
    internal const double AbsoluteTolerance = 1e-5;
    internal const double RelativeTolerance = 1e-4;

    public static int Run(string[] args)
    {
        Dictionary<string, string> options = [];
        for (int i = 0; i < args.Length; i += 2)
            if (i + 1 == args.Length || args[i] is not ("--reference" or "--mode" or "--profile" or "--case" or "--warmup") || !options.TryAdd(args[i], args[i + 1]))
                throw new ArgumentException("Invalid public-validation arguments.");
        string root = Path.GetFullPath(options.GetValueOrDefault("--reference") ?? throw new ArgumentException("Missing --reference."));
        string profile = options.GetValueOrDefault("--profile", "canonical"), mode = options.GetValueOrDefault("--mode", "both");
        int warmup = int.Parse(options.GetValueOrDefault("--warmup", "30"), System.Globalization.CultureInfo.InvariantCulture);
        if (warmup < 0) throw new ArgumentException("Public preflight warmup must be nonnegative.");
        if (mode is not ("both" or "rec_public_owned" or "ocr_owned")) throw new ArgumentException("Public validation requires REC, OCR or both.");
        OcrBenchmark.RequireProfile(profile);
        byte[] manifestBytes = File.ReadAllBytes(Path.Combine(root, "manifest.json"));
        using JsonDocument manifest = JsonDocument.Parse(manifestBytes);
        JsonElement document = manifest.RootElement, contract = document.GetProperty("contract");
        string family = contract.GetProperty("model").GetString()!;
        _ = Program.ModelPreset.For(family);
        if (document.GetProperty("schema_version").GetInt32() != 1 || contract.GetProperty("profile").GetString() != "canonical" || contract.GetProperty("capture_scope").GetString() != "ocr")
            throw new InvalidDataException("Public preflight requires a canonical full-OCR reference.");
        PaddleOcrOptions effectiveOptions = Program.ReadOptions(contract);
        Dictionary<string, JsonElement> assets = document.GetProperty("models").EnumerateArray().ToDictionary(a => a.GetProperty("id").GetString()!);
        foreach (string role in new[] { "det", "cls", "rec", "dictionary" })
            if (!assets.ContainsKey(role)) throw new InvalidDataException("Missing required full-OCR asset: " + role);
        foreach (JsonElement asset in assets.Values) _ = Program.ReadBundleFile(root, asset);
        foreach (JsonElement asset in document.GetProperty("auxiliary_files").EnumerateArray()) _ = Program.ReadBundleFile(root, asset);
        JsonElement[] cases = document.GetProperty("cases").EnumerateArray().ToArray();
        // Validate the whole immutable bundle before selection, matching the timing reader's gate.
        foreach (JsonElement item in cases)
            foreach (JsonElement artifact in item.GetProperty("artifacts").EnumerateArray()) _ = Program.ReadBundleFile(root, artifact);
        Model Load(string id)
        {
            using FileStream stream = File.OpenRead(BundleWriter.DestinationPath(root, assets[id].GetProperty("path").GetString()!));
            return Model.Load(stream);
        }
        bool includeRec = mode is "both" or "rec_public_owned", includeOcr = mode is "both" or "ocr_owned";
        using Model rec = Load("rec");
        byte[] dictionary = Program.ReadBundleFile(root, assets["dictionary"]);
        using Model? det = includeOcr ? Load("det") : null;
        using Model? cls = includeOcr ? Load("cls") : null;
        using PaddleOcrRecognizer recognizer = new(rec, dictionary, effectiveOptions.Recognizer);
        using PaddleOcrAll? ocr = includeOcr ? new(det!, cls!, rec, dictionary, effectiveOptions) : null;
        cases = cases.Where(c => !options.TryGetValue("--case", out string? selected) || c.GetProperty("id").GetString() == selected).ToArray();
        if (cases.Length == 0) throw new InvalidDataException("Unknown or empty public-validation case selection.");
        List<object> checks = [];
        int recognitionChecks = 0, ocrChecks = 0;
        foreach (JsonElement item in cases)
        {
            string caseId = item.GetProperty("id").GetString()!;
            Dictionary<string, JsonElement> artifacts = item.GetProperty("artifacts").EnumerateArray().ToDictionary(a => a.GetProperty("id").GetString()!);
            void Verify(string id, JsonElement input, object actual, string phase)
            {
                JsonElement expectedArtifact = artifacts[id];
                if (expectedArtifact.GetProperty("dtype").GetString() != "json") throw new InvalidDataException("Canonical public artifact must be JSON.");
                using JsonDocument expected = JsonDocument.Parse(Program.ReadBundleFile(root, expectedArtifact));
                byte[] actualBytes = BundleWriter.SerializeJson(actual);
                using JsonDocument actualDocument = JsonDocument.Parse(actualBytes);
                Compare(expected.RootElement, actualDocument.RootElement, caseId + "/" + id);
                checks.Add(new { caseId, artifactId = id, phase, inputArtifactId = input.GetProperty("id").GetString(), inputSha256 = input.GetProperty("sha256").GetString(),
                    expectedSha256 = expectedArtifact.GetProperty("sha256").GetString(), actualSha256 = BundleWriter.Hash(actualBytes), actual = actualDocument.RootElement.Clone() });
            }
            if (includeRec)
                foreach (var (id, input) in artifacts.Where(entry => entry.Key.EndsWith(".rec.pixels", StringComparison.Ordinal)))
                {
                    int[] shape = input.GetProperty("shape").EnumerateArray().Select(n => n.GetInt32()).ToArray();
                    byte[] pixels = Program.ReadBundleFile(root, input);
                    if (input.GetProperty("dtype").GetString() != "u8" || input.GetProperty("layout").GetString() != "bgr24" || shape.Length != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != 3 || pixels.Length != checked(shape[0] * shape[1] * 3))
                        throw new InvalidDataException("Invalid frozen oriented recognition pixels.");
                    // Frozen REC pixels are already oriented: no second classification or rotation.
                    PaddleOcrRecognitionResult result = recognizer.Recognize(pixels, shape[1], shape[0], true);
                    string publicId = id[..^6] + "public";
                    Verify(publicId, input, Program.RecognitionJson(result), "first_call");
                    for (int i = 0; i < warmup; i++) _ = recognizer.Recognize(pixels, shape[1], shape[0], true);
                    result = recognizer.Recognize(pixels, shape[1], shape[0], true);
                    Verify(publicId, input, Program.RecognitionJson(result), "after_warmup");
                    recognitionChecks++;
                }
            if (includeOcr)
            {
                using JsonDocument metadata = JsonDocument.Parse(Program.ReadBundleFile(root, artifacts["input.metadata"]));
                JsonElement info = metadata.RootElement;
                int width = info.GetProperty("width").GetInt32(), height = info.GetProperty("height").GetInt32(), stride = info.GetProperty("stride").GetInt32();
                JsonElement input = artifacts["input.pixels"];
                byte[] pixels = Program.ReadBundleFile(root, input);
                int[] shape = input.GetProperty("shape").EnumerateArray().Select(n => n.GetInt32()).ToArray();
                if (info.GetProperty("pixel_format").GetString() != "bgr24" || input.GetProperty("dtype").GetString() != "u8" || width <= 0 || height <= 0 || stride < checked(width * 3) || shape.Length != 2 || shape[0] != height || shape[1] != stride || pixels.Length < checked((height - 1) * stride + width * 3))
                    throw new InvalidDataException("Invalid frozen OCR pixels or stride.");
                PaddleOcrResult result = ocr!.Run(pixels, width, height, true, stride);
                Verify("result.public", input, Program.PublicOcrJson(result), "first_call");
                for (int i = 0; i < warmup; i++) _ = ocr.Run(pixels, width, height, true, stride);
                result = ocr.Run(pixels, width, height, true, stride);
                Verify("result.public", input, Program.PublicOcrJson(result), "after_warmup");
                ocrChecks++;
            }
        }
        if ((includeRec && recognitionChecks == 0) || (includeOcr && ocrChecks == 0)) throw new InvalidDataException("Public preflight did not execute every requested scope.");
        Console.WriteLine(Encoding.UTF8.GetString(BundleWriter.SerializeJson(new {
            schemaVersion = 1, kind = "ocr_public_semantic_validation", status = "passed", modelFamily = family, profile, mode,
            referenceProfile = "canonical", manifestSha256 = BundleWriter.Hash(manifestBytes), effectiveOptions,
            models = assets.Values.ToArray(), checkedCases = cases.Length, recognitionChecks, ocrChecks, warmup,
            recognitionCalls = recognitionChecks * (warmup + 2L), ocrCalls = ocrChecks * (warmup + 2L),
            runtimeControls = new[] { "DOTNET_PROCESSOR_COUNT", "DOTNET_EnableHWIntrinsic", "DOTNET_TieredCompilation", "DOTNET_TieredPGO", "PPOCR_NHWC", "SIMD_OCR_NOFUSE", "PPOCR_DET_THREADS" }
                .ToDictionary(key => key, Environment.GetEnvironmentVariable),
            atol = AbsoluteTolerance, rtol = RelativeTolerance,
            semantics = "Ordinary public CPU calls without trace hooks. Field sets, text, ordering, dimensions, labels, rotations, emitted counts and CTC bounds exact; scores and box coordinates use canonical-reference-magnitude tolerance. Compact raw-max scores are not probabilities.",
            framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            executableSha256 = BundleWriter.Hash(File.ReadAllBytes(typeof(PublicBenchmarkValidation).Assembly.Location)),
            csharpNativeSha256 = BundleWriter.Hash(File.ReadAllBytes(typeof(Model).Assembly.Location)), checks
        })));
        return 0;
    }

    internal static void Compare(JsonElement expected, JsonElement actual, string location, bool floating = false)
    {
        void Fail(string detail) => throw new InvalidDataException($"Public semantic mismatch at {location}: {detail}");
        if (expected.ValueKind != actual.ValueKind) Fail("JSON kinds differ");
        switch (expected.ValueKind)
        {
            case JsonValueKind.Object:
                Dictionary<string, JsonElement> Properties(JsonElement value)
                {
                    Dictionary<string, JsonElement> result = [];
                    foreach (JsonProperty field in value.EnumerateObject()) if (!result.TryAdd(field.Name, field.Value)) Fail("duplicate field " + field.Name);
                    return result;
                }
                var expectedFields = Properties(expected); var actualFields = Properties(actual);
                if (!expectedFields.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(actualFields.Keys)) Fail("field sets differ");
                foreach (var (name, value) in expectedFields)
                    Compare(value, actualFields[name], location + "." + name,
                        name is "score" or "recognition_score" or "classification_score" || (name == "points" && location.EndsWith(".box", StringComparison.Ordinal)));
                break;
            case JsonValueKind.Array:
                if (expected.GetArrayLength() != actual.GetArrayLength()) Fail("array lengths differ");
                for (int i = 0; i < expected.GetArrayLength(); i++) Compare(expected[i], actual[i], location + $"[{i}]", floating);
                break;
            case JsonValueKind.Number:
                if (!floating)
                {
                    if (!expected.TryGetInt64(out long expectedInteger) || !actual.TryGetInt64(out long actualInteger) || expectedInteger != actualInteger) Fail("integer field differs");
                }
                else
                {
                    double reference = expected.GetDouble(), candidate = actual.GetDouble();
                    if (!double.IsFinite(reference) || !double.IsFinite(candidate) || Math.Abs(reference - candidate) > AbsoluteTolerance + RelativeTolerance * Math.Abs(reference)) Fail("floating field exceeds declared tolerance");
                }
                break;
            case JsonValueKind.String:
                if (!string.Equals(expected.GetString(), actual.GetString(), StringComparison.Ordinal)) Fail("text differs");
                break;
            case JsonValueKind.Null:
            case JsonValueKind.True:
            case JsonValueKind.False:
                break;
            default:
                Fail("unsupported JSON kind");
                break;
        }
    }
}
