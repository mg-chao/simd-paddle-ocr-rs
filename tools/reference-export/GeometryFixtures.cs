using System.Reflection;
using Sdcb.SimdPaddleOCR;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Small authoritative pixel/geometry fixtures; no model or image decoder.</summary>
internal static class GeometryFixtures
{
    public static int Export(string[] args)
    {
        if (args.Length != 2 || args[0] != "--out") throw new ArgumentException("Use --export-geometry --out <new-directory>.");
        if (Environment.GetEnvironmentVariable("DOTNET_EnableHWIntrinsic") != "0")
            throw new InvalidOperationException("Geometry goldens require DOTNET_EnableHWIntrinsic=0 before process startup.");
        BundleWriter writer = new(args[1]);
        List<CaseEntry> cases = [];
        foreach ((int w, int h, int dw, int dh) in new[] { (28, 11, 160, 80), (1, 1, 32, 32), (30, 20, 64, 32), (120, 20, 160, 80), (37, 29, 64, 32) })
            foreach (ImagePixelFormat format in Enum.GetValues<ImagePixelFormat>())
                Resize(writer, cases, w, h, dw, dh, format);

        Map(writer, cases, "rectangle", 40, 32, (x, y) => x >= 5 && x <= 30 && y >= 7 && y <= 23 ? .9f : .1f);
        Map(writer, cases, "hollow", 48, 40, (x, y) => x >= 5 && x <= 40 && y >= 6 && y <= 33 && (x <= 8 || x >= 37 || y <= 9 || y >= 30) ? .9f : .1f, boxThreshold: 0f);
        Map(writer, cases, "diagonal", 40, 40, (x, y) => (x >= 5 && x <= 14 && y >= 5 && y <= 14) || (x >= 15 && x <= 24 && y >= 15 && y <= 24) ? .9f : .1f);
        Map(writer, cases, "diamond45", 40, 40, (x, y) => Math.Abs(x - 20) + Math.Abs(y - 20) <= 12 ? .9f : .1f);
        Map(writer, cases, "border", 40, 32, (x, y) => x <= 25 && y <= 20 ? .9f : .1f);
        Map(writer, cases, "dilation", 40, 32, (x, y) => y >= 7 && y <= 23 && ((x >= 4 && x <= 14) || (x >= 16 && x <= 30)) ? .9f : .1f, dilation: true);
        Map(writer, cases, "threshold-equality", 40, 32, (x, y) => x >= 5 && x <= 30 && y >= 7 && y <= 23 ? .2f : .1f);
        float above = BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(.2f) + 1);
        Map(writer, cases, "threshold-next-bit", 40, 32, (x, y) => x >= 5 && x <= 30 && y >= 7 && y <= 23 ? above : .1f, boxThreshold: 0f);
        Map(writer, cases, "candidate-cap", 48, 48, (x, y) => x % 16 >= 3 && x % 16 <= 12 && y % 16 >= 3 && y % 16 <= 12 ? .6f : .1f, candidates: 2);
        Map(writer, cases, "rejected-candidate-cap", 48, 48, (x, y) => x % 16 >= 3 && x % 16 <= 12 && y % 16 >= 3 && y % 16 <= 12 ? (y < 16 ? .3f : .9f) : .1f, candidates: 2);
        Map(writer, cases, "blank", 32, 32, (_, _) => .1f);

        Crop(writer, cases, "perspective", new(3, 4, 32, 5, 31, 24, 4, 23, 1));
        Crop(writer, cases, "vertical", new(3, 2, 11, 2, 11, 26, 3, 26, 1));
        Crop(writer, cases, "vertical-equality", new(3, 2, 11, 2, 11, 14, 3, 14, 1));
        Crop(writer, cases, "diamond45", new(18, 2, 31, 15, 18, 28, 5, 15, 1));
        Crop(writer, cases, "replicated-border", new(-2, -1, 25, 1, 28, 25, -1, 27, 1));
        Crop(writer, cases, "floor-dimensions", new(3, 2, 11.9f, 2, 11.9f, 13.9f, 3, 13.9f, 1));

        List<PaddleOcrDetectionBox> ties = [];
        for (int i = 0; i < 33; i++) ties.Add(new(10, 20, 10 + i + 5, 20, 10 + i + 5, 25, 10, 25, i / 32f));
        PaddleOcrDetectionBox[] initial = ties.ToArray();
        typeof(DbPostprocess).GetMethod("SortReadingOrder", BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, [ties]);
        string tieDirectory = "cases/sort-ties";
        cases.Add(new("sort-ties", "reading_order", [
            writer.WriteJson("fixture", $"{tieDirectory}/fixture.json", new { input = initial.Select(Box).ToArray() }),
            writer.WriteJson("boxes", $"{tieDirectory}/boxes.json", ties.Select(Box).ToArray())
        ]));

        writer.WriteManifest(new { schemaVersion = 1, contract = new { profile = "canonical", backend = "cpu", logicalInputLayout = "nchw", fixtureKind = "geometry-v1", decoder = "none; deterministic generated interleaved pixels", arithmetic = "serial original C# source, hardware intrinsics disabled" }, models = Array.Empty<ModelAsset>(), cases, auxiliaryFiles = Array.Empty<object>(), provenance = new { implementation = "original Sdcb.SimdPaddleOCR geometry/preprocessing", source = "GeometryFixtures.cs", framework = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription } });
        Console.WriteLine($"Geometry goldens written: {cases.Count} cases -> {writer.RootPath}");
        return 0;
    }

    private static byte[] Pixels(int width, int height, ImagePixelFormat format, int pad, out int stride)
    {
        int bpp = format is ImagePixelFormat.Bgr24 or ImagePixelFormat.Rgb24 ? 3 : 4;
        stride = width * bpp + pad;
        byte[] bytes = new byte[(height - 1) * stride + width * bpp];
        Array.Fill(bytes, (byte)243);
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
        {
            int source = (y * width + x) * 3, target = y * stride + x * bpp;
            byte b = (byte)((source * 73 + 19) % 256), g = (byte)(((source + 1) * 73 + 19) % 256), r = (byte)(((source + 2) * 73 + 19) % 256);
            bool swap = format is ImagePixelFormat.Rgb24 or ImagePixelFormat.Rgba32;
            bytes[target] = swap ? r : b; bytes[target + 1] = g; bytes[target + 2] = swap ? b : r;
            if (bpp == 4) bytes[target + 3] = (byte)((x * 17 + y * 31) % 256);
        }
        return bytes;
    }

    private static void Resize(BundleWriter writer, List<CaseEntry> cases, int w, int h, int dw, int dh, ImagePixelFormat format)
    {
        string id = $"resize-{w}x{h}-{format.ToString().ToLowerInvariant()}", directory = $"cases/{id}";
        byte[] source = Pixels(w, h, format, 13, out int stride);
        float[] det = new float[3 * dw * dh], cls = new float[3 * 80 * 160];
        PPOCRPreprocess.Det(source, w, h, stride, dw, dh, det, format);
        int contentWidth = PPOCRPreprocess.Cls(source, w, h, stride, cls, format);
        cases.Add(new(id, "resize", [
            writer.WriteJson("fixture", $"{directory}/fixture.json", new { width = w, height = h, stride, pixelFormat = format.ToString().ToLowerInvariant(), detWidth = dw, detHeight = dh, clsContentWidth = contentWidth }),
            writer.WriteBytes("pixels", $"{directory}/pixels.bin", source, [source.Length], "row_major"),
            writer.WriteFloats("det.input", $"{directory}/det.f32", det, [1, 3, dh, dw], "nchw"),
            writer.WriteFloats("cls.input", $"{directory}/cls.f32", cls, [1, 3, 80, 160], "nchw")
        ]));
    }

    private static void Map(BundleWriter writer, List<CaseEntry> cases, string name, int w, int h, Func<int, int, float> probability, bool dilation = false, float boxThreshold = .4f, int candidates = 3000)
    {
        string id = $"db-{name}", directory = $"cases/{id}";
        float[] map = new float[w * h];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) map[y * w + x] = probability(x, y);
        PaddleOcrDetectorOptions options = new() { BitmapThreshold = .2f, BoxThreshold = boxThreshold, UseDilation = dilation, UnclipRatio = 1.4f, MaxCandidates = candidates };
        PaddleOcrDetectionBox[] boxes = DbPostprocess.Run(map, w, h, options, w, h, 1, 1, new());
        cases.Add(new(id, "db", [
            writer.WriteJson("fixture", $"{directory}/fixture.json", new { width = w, height = h, sourceWidth = w, sourceHeight = h, options = new { bitmapThreshold = options.BitmapThreshold, boxThreshold, useDilation = dilation, unclipRatio = options.UnclipRatio, maxCandidates = candidates } }),
            writer.WriteFloats("db.map", $"{directory}/map.f32", map, [h, w], "row_major"),
            writer.WriteJson("boxes", $"{directory}/boxes.json", boxes.Select(Box).ToArray())
        ]));
    }

    private static void Crop(BundleWriter writer, List<CaseEntry> cases, string name, PaddleOcrDetectionBox box)
    {
        foreach (ImagePixelFormat format in Enum.GetValues<ImagePixelFormat>())
        {
            string id = $"crop-{name}-{format.ToString().ToLowerInvariant()}", directory = $"cases/{id}";
            byte[] source = Pixels(37, 29, format, 13, out int stride);
            byte[] crop = PPOCRCrop.Extract(source, 37, 29, stride, box, out int width, out int height, format);
            byte[] rotated = (byte[])crop.Clone(); PPOCRCrop.Rotate180(rotated, width, height);
            cases.Add(new(id, "crop", [
                writer.WriteJson("fixture", $"{directory}/fixture.json", new { width = 37, height = 29, stride, pixelFormat = format.ToString().ToLowerInvariant(), box = Box(box), cropWidth = width, cropHeight = height }),
                writer.WriteBytes("pixels", $"{directory}/pixels.bin", source, [source.Length], "row_major"),
                writer.WriteBytes("crop", $"{directory}/crop.bin", crop, [height, width, 3], "bgr24"),
                writer.WriteBytes("rotated", $"{directory}/rotated.bin", rotated, [height, width, 3], "bgr24")
            ]));
        }
    }
    private static object Box(PaddleOcrDetectionBox box) => new { points = new[] { new[] { box.X1, box.Y1 }, new[] { box.X2, box.Y2 }, new[] { box.X3, box.Y3 }, new[] { box.X4, box.Y4 } }, box.Score };
}
