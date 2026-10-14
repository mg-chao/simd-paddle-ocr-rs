using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace SimdPaddleOCR.ReferenceExport;

public sealed record CapturedFile(string SourcePath, string Path, string Sha256, long Bytes);
public sealed record CapturedTree(string TreeSha256, string TreeDigestFormat, List<CapturedFile> Files);
public sealed record GitProvenance(string Head, bool Dirty, string Porcelain, CapturedFile? TrackedDiff, string? SubmoduleStatus);
public sealed record AssemblyProvenance(string Name, string Path, string Sha256, long Bytes,
    string? Version = null, string? InformationalVersion = null);
public sealed record EnvironmentProvenance(string DotnetSdk, string Framework, string RuntimeVersion,
    string OperatingSystem, string OsArchitecture, string ProcessArchitecture, string? CpuIdentifier,
    int ProcessorCount, SortedDictionary<string, bool> CpuFeatures, SortedDictionary<string, string?> Variables,
    List<AssemblyProvenance> Assemblies);
public sealed record ReferenceProvenance(GitProvenance Git, CapturedTree SourceInputs, CapturedTree ExporterInputs,
    EnvironmentProvenance Environment, CapturedFile? Launcher = null);

/// <summary>Captures the actual source and runtime behind the reference, including an uncommitted checkout.</summary>
public static class Provenance
{
    private static readonly HashSet<string> SourceExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".cs", ".csproj", ".props", ".targets", ".slnx",
    };

    // Read only explicitly named runtime switches. Project-specific names are additionally
    // discovered from quoted literals in the captured source, never from an environment dump.
    private static readonly string[] RuntimeVariableNames =
    [
        "DOTNET_PROCESSOR_COUNT",
        "DOTNET_EnableHWIntrinsic", "DOTNET_EnableAVX", "DOTNET_EnableAVX2", "DOTNET_EnableAVX512",
        "DOTNET_EnableAVX512F", "DOTNET_EnableAVX512BW", "DOTNET_EnableAVX512CD", "DOTNET_EnableAVX512DQ",
        "DOTNET_EnableAVX512VL", "DOTNET_EnableAVX512VBMI", "DOTNET_EnableFMA", "DOTNET_EnableSSE",
        "DOTNET_EnableSSE2", "DOTNET_EnableSSE3", "DOTNET_EnableSSSE3", "DOTNET_EnableSSE41",
        "DOTNET_EnableSSE42", "DOTNET_EnableAES", "DOTNET_EnableBMI1", "DOTNET_EnableBMI2",
        "DOTNET_EnableLZCNT", "DOTNET_EnablePCLMULQDQ", "DOTNET_EnablePOPCNT", "DOTNET_EnableArm64AdvSimd",
        "DOTNET_EnableArm64Aes", "DOTNET_EnableArm64Crc32", "DOTNET_EnableArm64Sha1", "DOTNET_EnableArm64Sha256",
        "DOTNET_EnableArm64Sve", "DOTNET_EnableArm64Sve2", "DOTNET_EnableVectorT128", "DOTNET_EnableVectorT256",
        "DOTNET_EnableVectorT512",
    ];

    private static readonly Regex RelevantVariable = new(
        "\"(DOTNET_Enable[A-Za-z0-9_]+|PPOCR_[A-Za-z0-9_]+|SIMD_OCR_[A-Za-z0-9_]+)\"",
        RegexOptions.CultureInvariant);

    public static object Capture(string referenceRepo, string exporterSourceDirectory, string outputRoot)
    {
        string repo = Path.GetFullPath(referenceRepo);
        string exporter = Path.GetFullPath(exporterSourceDirectory);
        string output = Path.GetFullPath(outputRoot);
        if (!Directory.Exists(output)) throw new DirectoryNotFoundException($"Bundle output does not exist: {output}");
        if (!Directory.Exists(exporter)) throw new DirectoryNotFoundException($"Exporter sources do not exist: {exporter}");

        string actualRoot = RunProcess("git", repo, "rev-parse", "--show-toplevel").Trim();
        if (!string.Equals(Path.GetFullPath(actualRoot), repo.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            throw new ArgumentException("referenceRepo must be the Git repository root.", nameof(referenceRepo));
        string head = RunProcess("git", repo, "rev-parse", "HEAD").Trim();
        string porcelain = RunProcess("git", repo, "status", "--porcelain=v1", "--untracked-files=all");
        byte[] diff = RunProcessBytes("git", repo, "diff", "--no-ext-diff", "--no-textconv", "--binary",
            "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--");
        CapturedFile? diffFile = diff.Length == 0 ? null : StoreBytes(output, "HEAD diff", "provenance/reference.patch", diff);
        string submodules = RunProcess("git", repo, "submodule", "status", "--recursive").Trim();

        SortedSet<string> variableNames = new(RuntimeVariableNames, StringComparer.Ordinal);
        string[] sourcePaths = RunProcess("git", repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
            .Split('\0', StringSplitOptions.RemoveEmptyEntries)
            .Where(IsBuildInput)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();
        List<CapturedFile> sourceFiles = [];
        foreach (string relative in sourcePaths)
        {
            string source = BundleWriter.DestinationPath(repo, relative);
            // A tracked deletion is represented by the HEAD patch, not by a nonexistent snapshot.
            if (!File.Exists(source)) continue;
            byte[] bytes = File.ReadAllBytes(source);
            sourceFiles.Add(StoreBytes(output, relative.Replace('\\', '/'), $"provenance/source/{relative.Replace('\\', '/')}", bytes));
            DiscoverVariables(bytes, variableNames);
        }

        List<CapturedFile> exporterFiles = [];
        foreach (string source in Directory.EnumerateFiles(exporter, "*", SearchOption.AllDirectories)
            .Where(path => IsBuildInput(Path.GetRelativePath(exporter, path)))
            .OrderBy(path => Path.GetRelativePath(exporter, path).Replace('\\', '/'), StringComparer.Ordinal))
        {
            string relative = Path.GetRelativePath(exporter, source).Replace('\\', '/');
            byte[] bytes = File.ReadAllBytes(source);
            exporterFiles.Add(StoreBytes(output, relative, $"provenance/exporter/{relative}", bytes));
            DiscoverVariables(bytes, variableNames);
        }
        if (sourceFiles.Count == 0 || exporterFiles.Count == 0)
            throw new InvalidOperationException("The reference and exporter must both have captured build inputs.");

        string launcherPath = Path.Combine(Path.GetDirectoryName(exporter)!, "export-reference.ps1");
        CapturedFile? launcher = File.Exists(launcherPath)
            ? StoreBytes(output, "tools/export-reference.ps1", "provenance/exporter-launcher/export-reference.ps1",
                File.ReadAllBytes(launcherPath))
            : null;

        SortedDictionary<string, string?> variables = new(StringComparer.Ordinal);
        foreach (string name in variableNames) variables[name] = System.Environment.GetEnvironmentVariable(name);
        EnvironmentProvenance environment = new(
            RunProcess("dotnet", repo, "--version").Trim(),
            RuntimeInformation.FrameworkDescription,
            System.Environment.Version.ToString(),
            RuntimeInformation.OSDescription,
            RuntimeInformation.OSArchitecture.ToString(),
            RuntimeInformation.ProcessArchitecture.ToString(),
            System.Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER"),
            System.Environment.ProcessorCount,
            new(StringComparer.Ordinal)
            {
                ["adv_simd"] = AdvSimd.IsSupported,
                ["avx"] = Avx.IsSupported,
                ["avx2"] = Avx2.IsSupported,
                ["avx512f"] = Avx512F.IsSupported,
                ["fma"] = Fma.IsSupported,
                ["sse2"] = Sse2.IsSupported,
                ["sse41"] = Sse41.IsSupported,
            },
            variables,
            CaptureAssemblies());
        return new ReferenceProvenance(
            new(head, porcelain.Length != 0, porcelain, diffFile, submodules.Length == 0 ? null : submodules),
            Tree(sourceFiles), Tree(exporterFiles), environment, launcher);
    }

    private static bool IsBuildInput(string path)
    {
        string portable = path.Replace('\\', '/');
        string name = Path.GetFileName(portable);
        return (SourceExtensions.Contains(Path.GetExtension(portable))
            || name.Equals("global.json", StringComparison.OrdinalIgnoreCase)
            || name.Equals("packages.lock.json", StringComparison.OrdinalIgnoreCase)
            || name.Equals("NuGet.Config", StringComparison.OrdinalIgnoreCase))
            && !portable.Split('/').Any(segment => segment.Equals("bin", StringComparison.OrdinalIgnoreCase)
                || segment.Equals("obj", StringComparison.OrdinalIgnoreCase));
    }

    private static void DiscoverVariables(byte[] source, SortedSet<string> names)
    {
        foreach (Match match in RelevantVariable.Matches(Encoding.UTF8.GetString(source)))
            names.Add(match.Groups[1].Value);
    }

    private static CapturedFile StoreBytes(string output, string sourcePath, string path, byte[] bytes)
    {
        string relative = BundleWriter.NormalizeRelativePath(path);
        BundleWriter.WriteNewFile(output, relative, bytes);
        return new(sourcePath, relative, BundleWriter.Hash(bytes), bytes.LongLength);
    }

    private static CapturedTree Tree(List<CapturedFile> files)
    {
        files.Sort((left, right) => StringComparer.Ordinal.Compare(left.SourcePath, right.SourcePath));
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (CapturedFile file in files)
            hash.AppendData(Encoding.UTF8.GetBytes(file.SourcePath + "\0" + file.Sha256 + "\n"));
        return new(Convert.ToHexStringLower(hash.GetHashAndReset()),
            "sha256(UTF-8 concatenation of ordinal-sorted source_path + NUL + lowercase sha256 + LF)", files);
    }

    private static List<AssemblyProvenance> CaptureAssemblies()
    {
        IEnumerable<Assembly> assemblies = AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => !assembly.IsDynamic && (assembly == typeof(object).Assembly
                || assembly == typeof(Provenance).Assembly
                || assembly.GetName().Name?.Equals("SixLabors.ImageSharp", StringComparison.Ordinal) == true
                || assembly.GetName().Name?.StartsWith("Sdcb.SimdPaddleOCR", StringComparison.Ordinal) == true));
        List<AssemblyProvenance> result = [];
        foreach (Assembly assembly in assemblies.OrderBy(a => a.GetName().Name, StringComparer.Ordinal))
        {
            if (string.IsNullOrEmpty(assembly.Location) || !File.Exists(assembly.Location)) continue;
            using FileStream stream = File.OpenRead(assembly.Location);
            result.Add(new(assembly.GetName().Name!, assembly.Location,
                Convert.ToHexStringLower(SHA256.HashData(stream)), stream.Length,
                assembly.GetName().Version?.ToString(),
                assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion));
        }
        return result;
    }

    private static string RunProcess(string executable, string workingDirectory, params string[] arguments)
        => Encoding.UTF8.GetString(RunProcessBytes(executable, workingDirectory, arguments));

    private static byte[] RunProcessBytes(string executable, string workingDirectory, params string[] arguments)
    {
        ProcessStartInfo start = new(executable)
        {
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using Process process = Process.Start(start) ?? throw new InvalidOperationException($"Could not start {executable}.");
        using MemoryStream stdoutBytes = new();
        // Git patches can contain arbitrary text encodings. Preserve stdout bytes exactly.
        Task stdout = process.StandardOutput.BaseStream.CopyToAsync(stdoutBytes);
        Task<string> stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(30_000))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException($"{executable} did not complete within 30 seconds.");
        }
        Task.WhenAll(stdout, stderr).GetAwaiter().GetResult();
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"{executable} {string.Join(' ', arguments)} failed ({process.ExitCode}): {stderr.Result.Trim()}");
        return stdoutBytes.ToArray();
    }
}
