using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;

internal static class CharacterOracle
{
    internal static void Write(string output)
    {
        PaddleOcrDetectionBox horizontal = new(10, 20, 110, 20, 110, 40, 10, 40, .9f);
        PaddleOcrDetectionBox skewed = new(5, 13, 123, 7, 112, 45, 9, 52, .9f);
        PaddleOcrDetectionBox vertical = new(25, 5, 49, 8, 44, 163, 20, 150, .9f);
        PaddleOcrDetectionBox degenerate = new(0, 0, 0, 0, 0, 0, 0, 0, 1);
        PaddleOcrCtcSpan[] tokens = [new("中", .875f, 1, 3), new("a\u0301", .625f, 7, 9), new("🙂", .75f, 10, 11)];
        List<object> cases = [];
        void Capture(string id, PaddleOcrDetectionBox box, PaddleOcrCtcSpan[]? spans, int steps = 16, int content = 96, int tensor = 128, int rotation = 0)
        {
            object[]? expected = null;
            string? error = null;
            try
            {
                expected = PaddleOcrCharacterBoxes.Estimate(box, spans, steps, content, tensor, rotation)
                    .Select(character => (object)new { character.Text, character.Score, Points = new[] {
                        new[] {character.X1,character.Y1},new[] {character.X2,character.Y2},new[] {character.X3,character.Y3},new[] {character.X4,character.Y4} } }).ToArray();
            }
            catch (Exception exception) when (exception is ArgumentException or InvalidDataException or InvalidOperationException)
            { error = exception.GetType().Name; }
            cases.Add(new { Id = id, Detection = new { Points = new[] { new[] { box.X1, box.Y1 }, new[] { box.X2, box.Y2 }, new[] { box.X3, box.Y3 }, new[] { box.X4, box.Y4 } }, box.Score },
                Spans = spans, TimeSteps = steps, ContentWidth = content, TensorWidth = tensor, AppliedRotationDegrees = rotation, Expected = expected, Error = error });
        }
        Capture("padded-horizontal-gaps-unicode", horizontal, tokens);
        Capture("unpadded-horizontal", horizontal, tokens, content: 128);
        Capture("skewed-perspective", skewed, tokens);
        Capture("skewed-180", skewed, tokens, rotation: 180);
        Capture("vertical-crop", vertical, tokens);
        Capture("vertical-crop-180", vertical, tokens, rotation: 180);
        Capture("right-padding-clamps", horizontal, [new("x", .5f, 13, 16)], content: 48);
        Capture("touching-token-runs", horizontal, [new("a", .5f, 2, 6), new("b", .75f, 6, 9)]);
        Capture("empty-before-geometry-validation", degenerate, [], steps: 0, content: 0, tensor: 0, rotation: 90);
        Capture("missing-alignment", horizontal, null);
        Capture("overlapping-spans", horizontal, [new("a", .5f, 1, 7), new("b", .75f, 5, 9)]);
        Capture("empty-run", horizontal, [new("a", .5f, 4, 4)]);
        Capture("run-after-time-axis", horizontal, [new("a", .5f, 15, 17)]);
        Capture("zero-time-axis", horizontal, tokens, steps: 0);
        Capture("zero-content-width", horizontal, tokens, content: 0);
        Capture("tensor-smaller-than-content", horizontal, tokens, content: 129);
        Capture("unsupported-rotation", horizontal, tokens, rotation: 90);
        Capture("degenerate-detection", degenerate, tokens);
        object Pack(string id, params string[] lines)
        {
            PaddleOcrResult result = new() { DetectedCount = lines.Length, DetectorResizedWidth = 128,
                DetectorResizedHeight = 64, Lines = lines.Select(text => new PaddleOcrLine { Box = horizontal,
                    Text = text, RecognitionScore = .75f, ClassificationScore = 1f, ClassificationLabel = 0,
                    AppliedRotationDegrees = 0, EmittedCount = 0 }).ToArray() };
            return new { Id = id, Lines = lines, result.Text, result.PackedText,
                PackedUtf8Hex = Convert.ToHexStringLower(System.Text.Encoding.UTF8.GetBytes(result.PackedText)), result.PackedTextHash };
        }
        object[] packedCases = [Pack("no-lines"), Pack("empty-line", ""),
            Pack("unicode-tokens", "中", "a\u0301", "🙂"), Pack("embedded-nul-and-newline", "a\0b", "line\nbreak", ""),
            Pack("ordered-empty-lines", "", "中", "", "🙂"), Pack("multilingual", "Résumé", "Русский", "العربية")];
        var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
        Directory.CreateDirectory(output);
        string assembly = typeof(PaddleOcrCharacterBoxes).Assembly.Location;
        object data = new { SchemaVersion = 1, ReferenceRevision = "33ff8a81672452041d110e8ba52dcd2639e54228",
            Framework = RuntimeInformation.FrameworkDescription,
            ReferenceAssemblySha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(assembly))),
            Generator = "tools/reference-exporter/CharacterOracle.cs", Cases = cases, PackedTextCases = packedCases };
        File.WriteAllText(Path.Combine(output, "character-oracle.json"), JsonSerializer.Serialize(data, json));
        Console.WriteLine($"Captured {cases.Count} independent character-box and {packedCases.Length} packed-text contracts");
    }
}
