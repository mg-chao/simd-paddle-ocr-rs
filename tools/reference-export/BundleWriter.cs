using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SimdPaddleOCR.ReferenceExport;

public sealed record Artifact(string Id, string Path, string Dtype, int[] Shape, string Layout, string Sha256, long Bytes);
public sealed record ModelAsset(string Id, string Path, string Sha256, long Bytes);
public sealed record CaseEntry(string Id, string Kind, List<Artifact> Artifacts);

/// <summary>Writes an immutable, portable reference bundle. Existing bundles are never overwritten.</summary>
public sealed class BundleWriter
{
    public static JsonSerializerOptions JsonOptions { get; } = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public string RootPath { get; }

    public BundleWriter(string outputRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputRoot);
        RootPath = System.IO.Path.GetFullPath(outputRoot);
        if (Directory.Exists(RootPath) || File.Exists(RootPath))
            throw new IOException($"Output already exists: {RootPath}. Choose a new directory.");
        Directory.CreateDirectory(RootPath);
    }

    public Artifact WriteBytes(string id, string relativePath, ReadOnlySpan<byte> bytes, int[] shape, string layout)
    {
        ValidateArtifact(id, shape, layout, bytes.Length);
        string path = NormalizeRelativePath(relativePath);
        WriteNewFile(RootPath, path, bytes);
        return new(id, path, "u8", (int[])shape.Clone(), layout, Hash(bytes), bytes.Length);
    }

    public Artifact WriteFloats(string id, string relativePath, ReadOnlySpan<float> values, int[] shape, string layout)
    {
        ValidateArtifact(id, shape, layout, values.Length);
        for (int i = 0; i < values.Length; i++)
            if (!float.IsFinite(values[i]))
                throw new ArgumentException($"Artifact {id} contains a non-finite float at element {i}.", nameof(values));

        string path = NormalizeRelativePath(relativePath);
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using FileStream output = CreateNewFile(RootPath, path);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(64 * 1024);
        try
        {
            int offset = 0;
            while (offset < values.Length)
            {
                int count = Math.Min(values.Length - offset, buffer.Length / sizeof(float));
                Span<byte> bytes = buffer.AsSpan(0, count * sizeof(float));
                for (int i = 0; i < count; i++)
                    BinaryPrimitives.WriteSingleLittleEndian(bytes.Slice(i * sizeof(float), sizeof(float)), values[offset + i]);
                output.Write(bytes);
                hash.AppendData(bytes);
                offset += count;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
        return new(id, path, "f32", (int[])shape.Clone(), layout,
            Convert.ToHexStringLower(hash.GetHashAndReset()), (long)values.Length * sizeof(float));
    }

    public Artifact WriteJson(string id, string relativePath, object value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentNullException.ThrowIfNull(value);
        byte[] bytes = SerializeJson(value);
        string path = NormalizeRelativePath(relativePath);
        WriteNewFile(RootPath, path, bytes);
        return new(id, path, "json", [], "json", Hash(bytes), bytes.LongLength);
    }

    public ModelAsset CopyModel(string id, string sourcePath, string relativePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        string path = NormalizeRelativePath(relativePath);
        using FileStream input = File.OpenRead(sourcePath);
        using FileStream output = CreateNewFile(RootPath, path);
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(64 * 1024);
        long length = 0;
        try
        {
            int read;
            while ((read = input.Read(buffer)) != 0)
            {
                output.Write(buffer, 0, read);
                hash.AppendData(buffer, 0, read);
                length += read;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
        return new(id, path, Convert.ToHexStringLower(hash.GetHashAndReset()), length);
    }

    public void WriteManifest(object value) => WriteJson("manifest", "manifest.json", value);

    internal static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    internal static byte[] SerializeJson(object value)
    {
        JsonElement element = JsonSerializer.SerializeToElement(value, JsonOptions);
        using MemoryStream stream = new();
        using (Utf8JsonWriter writer = new(stream, new JsonWriterOptions { Indented = true }))
            WriteSortedJson(writer, element);
        stream.WriteByte((byte)'\n');
        return stream.ToArray();
    }

    private static void WriteSortedJson(Utf8JsonWriter writer, JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            writer.WriteStartObject();
            foreach (JsonProperty property in value.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal))
            {
                writer.WritePropertyName(property.Name);
                WriteSortedJson(writer, property.Value);
            }
            writer.WriteEndObject();
        }
        else if (value.ValueKind == JsonValueKind.Array)
        {
            writer.WriteStartArray();
            foreach (JsonElement item in value.EnumerateArray()) WriteSortedJson(writer, item);
            writer.WriteEndArray();
        }
        else value.WriteTo(writer);
    }

    private static void ValidateArtifact(string id, int[] shape, string layout, long count)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentException.ThrowIfNullOrWhiteSpace(layout);
        ArgumentNullException.ThrowIfNull(shape);
        long expected = 1;
        foreach (int dimension in shape)
        {
            if (dimension < 0) throw new ArgumentException("Tensor dimensions must be nonnegative.", nameof(shape));
            expected = checked(expected * dimension);
        }
        if (expected != count)
            throw new ArgumentException($"Artifact {id} shape describes {expected} elements, but received {count}.", nameof(shape));
    }

    internal static string NormalizeRelativePath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string portable = path.Replace('\\', '/');
        if (System.IO.Path.IsPathRooted(path) || portable.StartsWith('/'))
            throw new ArgumentException("Bundle paths must be relative.", nameof(path));
        foreach (string segment in portable.Split('/'))
        {
            if (segment.Length == 0 || segment is "." or ".." || segment.EndsWith('.') || segment.EndsWith(' ')
                || segment.Any(c => c < 32 || "<>:\"|?*".Contains(c)))
                throw new ArgumentException($"Unsafe or nonportable bundle path: {path}", nameof(path));
            string stem = segment.Split('.')[0];
            if (stem.Equals("CON", StringComparison.OrdinalIgnoreCase)
                || stem.Equals("PRN", StringComparison.OrdinalIgnoreCase)
                || stem.Equals("AUX", StringComparison.OrdinalIgnoreCase)
                || stem.Equals("NUL", StringComparison.OrdinalIgnoreCase)
                || (stem.Length == 4 && (stem.StartsWith("COM", StringComparison.OrdinalIgnoreCase)
                    || stem.StartsWith("LPT", StringComparison.OrdinalIgnoreCase)) && stem[3] is >= '1' and <= '9'))
                throw new ArgumentException($"Reserved device name in bundle path: {path}", nameof(path));
        }
        return portable;
    }

    internal static string DestinationPath(string root, string path)
    {
        string fullRoot = System.IO.Path.GetFullPath(root);
        string fullPath = System.IO.Path.GetFullPath(System.IO.Path.Combine(fullRoot,
            NormalizeRelativePath(path).Replace('/', System.IO.Path.DirectorySeparatorChar)));
        StringComparison comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (!fullPath.StartsWith(fullRoot.TrimEnd(System.IO.Path.DirectorySeparatorChar, System.IO.Path.AltDirectorySeparatorChar)
            + System.IO.Path.DirectorySeparatorChar, comparison))
            throw new ArgumentException("Bundle path escapes its root.", nameof(path));
        return fullPath;
    }

    internal static FileStream CreateNewFile(string root, string relativePath)
    {
        string path = DestinationPath(root, relativePath);
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);
        return new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
    }

    internal static void WriteNewFile(string root, string relativePath, ReadOnlySpan<byte> bytes)
    {
        using FileStream stream = CreateNewFile(root, relativePath);
        stream.Write(bytes);
    }
}
