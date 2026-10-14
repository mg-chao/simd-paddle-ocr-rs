using System.Text.Json;
using System.Text.Json.Nodes;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Model-free launcher/replay contract checks, separate from native graph parity.</summary>
internal static class ExportContractTests
{
    public static int Run()
    {
        string temporaryRoot = Path.GetFullPath(Path.GetTempPath());
        string root = Path.Combine(temporaryRoot, "simd-ocr-export-contracts-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        int checks = 0;
        try
        {
            string[] Required() => ["--reference-repo", root, "--out", Path.Combine(root, "output"), "--exporter-source", root];
            Program.Settings Parse(params string[] supplied) => Program.Settings.Parse(Required().Concat(supplied).ToArray());
            void Check(bool condition, string detail)
            {
                if (!condition) throw new InvalidDataException("Export contract regression: " + detail);
                checks++;
            }
            void Reject(string detail, params string[] supplied)
            {
                try { _ = Parse(supplied); }
                catch (ArgumentException) { checks++; return; }
                throw new InvalidDataException("Exporter accepted conflicting settings: " + detail);
            }
            foreach (string family in new[] { "tiny", "small", "medium" })
            {
                Program.Settings configured = Parse("--model", family);
                Check(configured.Model == family && configured.EffectiveOptions().Detector.BoxThreshold == (family == "tiny" ? .4f : .45f), family + " model defaults");
                Check(Program.ModelPreset.For(family).DictionaryFile == (family == "tiny" ? "ppocr_keys.txt" : "rec_keys.txt"), family + " local dictionary role");
            }
            JsonNode effective = JsonSerializer.SerializeToNode(Program.Options("small"), BundleWriter.JsonOptions)!;
            effective["detector"]!["box_threshold"] = .67f;
            effective["recognizer"]!["adaptive_width"] = false;
            effective["recognizer"]!["target_width"] = 96;
            File.WriteAllBytes(Path.Combine(root, "manifest.json"), BundleWriter.SerializeJson(new {
                schemaVersion = 1, contract = new { model = "small", captureScope = "ocr", profile = "canonical", nodes = "all", maxTracedLines = 0, options = effective } }));
            Program.Settings replay = Parse("--replay", root);
            Check(replay.Model == "small" && replay.Nodes == "all" && replay.MaxTracedLines == 0, "replay inherits family and trace contract");
            Check(replay.EffectiveOptions().Detector.BoxThreshold == .67f && !replay.EffectiveOptions().Recognizer.AdaptiveWidth && replay.EffectiveOptions().Recognizer.TargetWidth == 96, "replay retains effective options");
            Reject("family", "--replay", root, "--model", "medium");
            Reject("scope", "--replay", root, "--capture-scope", "step12");
            Reject("source mode", "--replay", root, "--input-bundle", root);
            Reject("input modes", "--input", root, "--input-bundle", root);
            Reject("OCR DetectorModel", "--model", "small", "--detector-model", "small");
            Reject("invalid family", "--model", "unsupported");
            Program.Settings fresh = Parse("--input-bundle", root, "--model", "medium");
            Check(fresh.Replay is null && fresh.InputBundle == root && fresh.EffectiveOptions().Detector.BoxThreshold == .45f && fresh.EffectiveOptions().Recognizer.AdaptiveWidth, "input-only reuse uses target defaults");
            effective["line_worker_count"] = 2;
            using JsonDocument invalid = JsonDocument.Parse(BundleWriter.SerializeJson(new { options = effective }));
            try { _ = Program.ReadOptions(invalid.RootElement); }
            catch (InvalidDataException) { checks++; Console.WriteLine($"Exporter model/replay contract regressions passed: {checks} checks."); return 0; }
            throw new InvalidDataException("Exporter accepted a multiworker replay contract.");
        }
        finally
        {
            string resolved = Path.GetFullPath(root);
            if (!resolved.StartsWith(temporaryRoot.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ||
                !Path.GetFileName(resolved).StartsWith("simd-ocr-export-contracts-", StringComparison.Ordinal))
                throw new InvalidOperationException("Temporary cleanup path is outside the exporter contract-test root.");
            Directory.Delete(resolved, true);
        }
    }
}
