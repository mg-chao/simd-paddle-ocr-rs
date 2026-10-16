using System.Buffers;
using System.Diagnostics.Tracing;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Reflection;
using System.Runtime.InteropServices;
using Sdcb.SimdPaddleOCR;

// This listener exists only during a separate diagnostic batch. It is disposed
// before any official latency/throughput sample or GC allocation-volume probe.
internal sealed class NumericPoolProbe : EventListener
{
    private sealed record Pool(string Name, int ElementBytes);
    private static readonly Type RunType = typeof(PaddleOcrRecognizer).GetNestedType("CtcRun", BindingFlags.NonPublic)
        ?? throw new MissingFieldException("Pinned CTC alignment scratch type changed");
    private static readonly Type RunPoolType = typeof(ArrayPool<>).MakeGenericType(RunType);
    private static readonly object RunPool = RunPoolType.GetProperty("Shared")!.GetValue(null)!;
    private readonly object _gate = new();
    private readonly Dictionary<int, Pool> _pools = new()
    {
        [RuntimeHelpers.GetHashCode(ArrayPool<float>.Shared)] = new("float", 4),
        [RuntimeHelpers.GetHashCode(ArrayPool<int>.Shared)] = new("int", 4),
        [RuntimeHelpers.GetHashCode(ArrayPool<uint>.Shared)] = new("uint", 4),
        [RuntimeHelpers.GetHashCode(ArrayPool<byte>.Shared)] = new("byte", 1),
        [RuntimeHelpers.GetHashCode(ArrayPool<short>.Shared)] = new("short", 2),
        [RuntimeHelpers.GetHashCode(ArrayPool<char>.Shared)] = new("char", 2),
        [RuntimeHelpers.GetHashCode(ArrayPool<Vector256<float>>.Shared)] = new("Vector256<float>", 32),
        [RuntimeHelpers.GetHashCode(ArrayPool<Vector512<float>>.Shared)] = new("Vector512<float>", 64),
        [RuntimeHelpers.GetHashCode(RunPool)] = new("CtcRun", Marshal.SizeOf(RunType)),
    };
    private readonly Dictionary<(int Pool, int Buffer), long> _live = [];
    private readonly Dictionary<string, string[]> _schemas = [];
    private readonly HashSet<string> _seenTypes = [];
    private EventSource? _source;
    private bool _active = true;
    private long _liveBytes, _peakBytes, _rentedBytes, _rents, _returns;
    private long _unmatchedReturns, _unmatchedReturnBytes;
    private string? _error;

    protected override void OnEventSourceCreated(EventSource source)
    {
        if (source.Name == "System.Buffers.ArrayPoolEventSource")
        {
            _source = source;
            EnableEvents(source, EventLevel.Verbose);
        }
    }

    protected override void OnEventWritten(EventWrittenEventArgs data)
    {
        // EventListener's base constructor can call virtual methods before the
        // derived object's field initializers have completed.
        if (_gate is null || _pools is null || data.EventName is not ("BufferRented" or "BufferReturned")) return;
        lock (_gate)
        {
            if (!_active || _error is not null) return;
            string[] expected = data.EventName == "BufferRented"
                ? ["bufferId", "bufferSize", "poolId", "bucketId"] : ["bufferId", "bufferSize", "poolId"];
            if (data.PayloadNames is null || !data.PayloadNames.SequenceEqual(expected)
                || data.Payload is null || data.Payload.Count != expected.Length || data.Payload.Any(value => value is not int))
            {
                _error = "ArrayPool event payload schema changed: " + data.EventName;
                return;
            }
            int buffer = (int)data.Payload[0]!, length = (int)data.Payload[1]!, poolId = (int)data.Payload[2]!;
            if (!_pools.TryGetValue(poolId, out Pool? pool)) return;
            _schemas.TryAdd(data.EventName, expected);
            if (length < 0) { _error = "Negative numeric ArrayPool length"; return; }
            var key = (poolId, buffer);
            if (data.EventName == "BufferRented")
            {
                long bytes = (long)length * pool.ElementBytes;
                if (!_live.TryAdd(key, bytes)) { _error = "Numeric ArrayPool buffer rented twice without return"; return; }
                _seenTypes.Add(pool.Name); _rents++; _rentedBytes += bytes; _liveBytes += bytes;
                _peakBytes = Math.Max(_peakBytes, _liveBytes);
            }
            else
            {
                if (!_live.Remove(key, out long bytes)) { _unmatchedReturns++; _unmatchedReturnBytes += (long)length * pool.ElementBytes; return; }
                if (bytes != (long)length * pool.ElementBytes) { _error = "Numeric ArrayPool return size changed"; return; }
                _returns++; _liveBytes -= bytes;
            }
        }
    }

    private object Finish(bool selfTest, long expectedSelfTestBytes = 0, object? discardBoundary = null)
    {
        if (_source is not null) DisableEvents(_source);
        lock (_gate)
        {
            _active = false;
            if (_error is not null) throw new InvalidDataException(_error);
            if (selfTest && (_liveBytes != 0 || _live.Count != 0 || _rents != _returns || _unmatchedReturns != 0
                || _rents != 9 || _peakBytes != expectedSelfTestBytes || _rentedBytes != expectedSelfTestBytes || _seenTypes.Count != 9 || _schemas.Count != 2))
                throw new InvalidDataException("The runtime did not expose the verified numeric ArrayPool event schema/pool identities");
            return new { Supported = true, SelfTest = selfTest, PeakObservedUnreturnedBytes = _peakBytes, RentedBytes = _rentedBytes,
                RentCount = _rents, ReturnCount = _returns, UnreturnedRentalBytes = _liveBytes,
                UnmatchedReturnCount = _unmatchedReturns, UnmatchedReturnBytes = _unmatchedReturnBytes,
                UnreturnedByType = _live.GroupBy(pair => _pools[pair.Key.Pool].Name).Select(group => new {
                    Type = group.Key, Count = group.Count(), Bytes = group.Sum(pair => pair.Value),
                    AllMeetDiscardCapacity = group.All(pair => pair.Value >= PooledArrays.DiscardThresholdBytes) }).ToArray(),
                NumericTypesObserved = _seenTypes.Order().ToArray(), EventSource = "System.Buffers.ArrayPoolEventSource", Schemas = _schemas,
                Pools = _pools.Select(pair => new { PoolId = pair.Key, pair.Value.Name, pair.Value.ElementBytes }).ToArray(),
                DiscardBoundarySelfTest = discardBoundary,
                Contract = "separate diagnostic process, one warmed batch of active CPU ArrayPool<float/int/uint/byte/short/char/Vector256<float>/Vector512<float>/CtcRun> rentals; counts actual array capacities. Peak observed unreturned capacity is a conservative event high-water mark, not exact live memory: PooledArrays may discard >=65536B buckets without Return events. Remaining rentals are observations, not leak proof. Excludes direct unpooled allocations, retained request arrays and fixed stack tiles. Official timing/allocation processes never enable this listener" };
        }
    }

    internal static object VerifySupport()
    {
        object boundary = VerifyDiscardBoundary();
        using NumericPoolProbe listener = new();
        float[] f = ArrayPool<float>.Shared.Rent(11);
        int[] i = ArrayPool<int>.Shared.Rent(13);
        uint[] u = ArrayPool<uint>.Shared.Rent(7);
        byte[] b = ArrayPool<byte>.Shared.Rent(5);
        short[] s = ArrayPool<short>.Shared.Rent(9);
        char[] c = ArrayPool<char>.Shared.Rent(3);
        Vector256<float>[] v256 = ArrayPool<Vector256<float>>.Shared.Rent(5);
        Vector512<float>[] v = ArrayPool<Vector512<float>>.Shared.Rent(3);
        Array runs = (Array)RunPoolType.GetMethod("Rent")!.Invoke(RunPool, [3])!;
        long expected = (long)f.Length * 4 + (long)i.Length * 4 + (long)u.Length * 4 + b.Length
            + (long)s.Length * 2 + (long)c.Length * 2 + (long)v256.Length * 32 + (long)v.Length * 64 + runs.Length * Marshal.SizeOf(RunType);
        ArrayPool<float>.Shared.Return(f); ArrayPool<int>.Shared.Return(i); ArrayPool<Vector512<float>>.Shared.Return(v);
        ArrayPool<uint>.Shared.Return(u); ArrayPool<byte>.Shared.Return(b); ArrayPool<short>.Shared.Return(s);
        ArrayPool<char>.Shared.Return(c); ArrayPool<Vector256<float>>.Shared.Return(v256); RunPoolType.GetMethod("Return")!.Invoke(RunPool, [runs, false]);
        return listener.Finish(true, expected, boundary);
    }

    private static object VerifyDiscardBoundary()
    {
        using NumericPoolProbe listener = new();
        int[] values = PooledArrays.Rent<int>(8193);
        PooledArrays.Return(values);
        object measurements = listener.Finish(false);
        long actual = (long)values.Length * 4;
        if (actual < PooledArrays.DiscardThresholdBytes || listener._rents != 1 || listener._returns != 0
            || listener._liveBytes != actual || listener._unmatchedReturns != 0)
            throw new InvalidDataException("Pinned pool discard boundary no longer matches the diagnostic accounting");
        return new { Supported = true, RequestedElements = 8193, ActualElements = values.Length,
            UnreturnedCapacityBytes = actual, ExpectedNoReturnEvent = true, Measurements = measurements,
            Contract = "valid PooledArrays discard: requested int bytes below 65536, actual pool bucket capacity at least 65536; no Return event is emitted" };
    }

    internal static object Measure(Action operation)
    {
        using NumericPoolProbe listener = new();
        operation();
        return listener.Finish(false);
    }
}
