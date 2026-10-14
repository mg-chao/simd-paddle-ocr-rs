using System.Security.Cryptography;
using Sdcb.SimdPaddleOCR;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Independent native REC resize/normalization boundary goldens.</summary>
internal static class RecFixtures
{
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
