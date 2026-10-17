using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

// Native counters are shared with the Rust worker's measurement contract.
// Darwin ru_maxrss is bytes, not the KiB used on Linux or Windows private commit.
static class ProcessMetrics
{
    public static bool SupportedHost =>
        (OperatingSystem.IsWindows() && RuntimeInformation.ProcessArchitecture == Architecture.X64) ||
        (OperatingSystem.IsMacOS() && Environment.Is64BitProcess);

    public static void Configure(Process process, string affinity, int threads)
    {
        if (OperatingSystem.IsWindows())
        {
            ulong mask = Convert.ToUInt64(affinity, 16);
            if (BitOperations.PopCount(mask) != threads)
                throw new InvalidDataException("affinity CPU count must equal thread budget");
            process.ProcessorAffinity = unchecked((IntPtr)(long)mask);
        }
        else if (!OperatingSystem.IsMacOS() || affinity != "unbound")
            throw new InvalidDataException("macOS benchmark jobs must declare unbound scheduling");
    }

    public static TimeSpan ProcessCpuTime(Process process)
    {
        if (OperatingSystem.IsWindows()) return process.TotalProcessorTime;
        Usage usage = ReadUsage();
        static long Microseconds(TimeValue value)
        {
            if (value.Seconds < 0 || value.Microseconds is < 0 or >= 1_000_000)
                throw new InvalidDataException("invalid process CPU time");
            return checked(value.Seconds * 1_000_000 + value.Microseconds);
        }
        return TimeSpan.FromTicks(checked((Microseconds(usage.User) + Microseconds(usage.System)) * 10));
    }

    public static (ulong? PeakPrivateBytes, ulong? PeakResidentBytes) PeakMemory()
    {
        if (OperatingSystem.IsWindows())
        {
            Counters counters = new() { Size = (uint)Marshal.SizeOf<Counters>() };
            using Process process = Process.GetCurrentProcess();
            if (!GetProcessMemoryInfo(process.Handle, ref counters, counters.Size))
                throw new System.ComponentModel.Win32Exception(Marshal.GetLastPInvokeError());
            return ((ulong)counters.PeakPagefile, null);
        }
        long peak = ReadUsage().PeakResident;
        if (peak <= 0) throw new InvalidDataException("invalid resident high-water mark");
        return (null, checked((ulong)peak));
    }

    public static object Probe()
    {
        if (!SupportedHost) throw new PlatformNotSupportedException();
        using Process process = Process.GetCurrentProcess();
        var before = ProcessCpuTime(process);
        var memory = PeakMemory();
        var after = ProcessCpuTime(process);
        if (after < before || (memory.PeakPrivateBytes ?? memory.PeakResidentBytes) is not > 0)
            throw new InvalidDataException("invalid native process counters");
        if (OperatingSystem.IsMacOS() && Marshal.SizeOf<Usage>() != 144)
            throw new InvalidDataException("unexpected Darwin rusage ABI");
        return new { CpuBeforeMs = before.TotalMilliseconds, CpuAfterMs = after.TotalMilliseconds,
            memory.PeakPrivateBytes, memory.PeakResidentBytes,
            DarwinRusageBytes = OperatingSystem.IsMacOS() ? Marshal.SizeOf<Usage>() : 0 };
    }

    static Usage ReadUsage()
    {
        if (!OperatingSystem.IsMacOS()) throw new PlatformNotSupportedException();
        if (GetResourceUsage(0, out Usage usage) != 0)
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastPInvokeError());
        return usage;
    }

    // 64-bit Darwin SDK sys/resource.h: two 16-byte timevals, fourteen longs.
    [StructLayout(LayoutKind.Sequential)]
    struct TimeValue { public long Seconds; public int Microseconds; }
    [StructLayout(LayoutKind.Sequential)]
    struct Usage
    {
        public TimeValue User, System;
        public long PeakResident, Shared, Data, Stack, MinorFaults, MajorFaults, Swaps,
            Inputs, Outputs, Sent, Received, Signals, Voluntary, Involuntary;
    }
    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "getrusage", SetLastError = true)]
    static extern int GetResourceUsage(int who, out Usage usage);

    [StructLayout(LayoutKind.Sequential)]
    struct Counters
    {
        public uint Size, Faults;
        public nuint PeakWorkingSet, WorkingSet, PeakPagedPool, PagedPool, PeakNonpagedPool, NonpagedPool,
            Pagefile, PeakPagefile, Private;
    }
    [DllImport("psapi.dll", SetLastError = true)]
    static extern bool GetProcessMemoryInfo(IntPtr process, ref Counters counters, uint size);
}
