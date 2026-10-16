"""Untimed Windows CPU contention check; records names/PIDs, never command lines.

CPU load is expressed in occupied logical cores, not percent of the whole host.
The policy permits light desktop activity; integrity and compiler-exit guards
remain conservative. Scheduling and retries belong to the acceptance driver.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import os
import time


POLICY = {
    "revision": "normal-background-v3",
    "description": "Allows normal desktop activity while rejecting heavy builds and high aggregate load",
    "compiler_core_limit": 0.50,
    "external_process_core_limit": 3.00,
    "aggregate_external_core_limit": 6.00,
    "unattributed_core_limit": 3.00,
    "units": "occupied logical cores over the sampling interval",
    "retry": "none; contention fails the performance run",
    "processor_groups": "idle proof requires at most 64 active logical processors",
}
COMPILERS = frozenset((
    "rustc.exe", "cargo.exe", "link.exe", "cl.exe", "clang.exe", "clang-cl.exe",
    "lld.exe", "lld-link.exe", "msbuild.exe", "dotnet.exe", "csc.exe", "vbc.exe",
    "ninja.exe", "cmake.exe", "tracker.exe", "rc.exe", "mspdbsrv.exe",
))


def _filetime(value) -> int:
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


def _snapshot() -> dict:
    if os.name != "nt":
        raise RuntimeError("CPU contention sampling requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
        function.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    pointers = [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *pointers]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetSystemTimes.argtypes = pointers[:3]
    kernel.GetSystemTimes.restype = wintypes.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.GetActiveProcessorCount.argtypes = [wintypes.WORD]
    kernel.GetActiveProcessorCount.restype = wintypes.DWORD
    processors = kernel.GetActiveProcessorCount(0xffff)  # ALL_PROCESSOR_GROUPS
    if not processors:
        raise ctypes.WinError(ctypes.get_last_error())

    def own_cpu_seconds():
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(value) for value in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (_filetime(values[2]) + _filetime(values[3])) / 10_000_000

    own_before = own_cpu_seconds()
    idle, system, user = (wintypes.FILETIME() for _ in range(3))
    if not kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())
    captured_at = time.monotonic()
    own_after = own_cpu_seconds()
    busy_seconds = (_filetime(system) + _filetime(user) - _filetime(idle)) / 10_000_000
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        present = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        if not present:
            raise ctypes.WinError(ctypes.get_last_error())
        while present:
            pid = entry.th32ProcessID
            if pid > 4:
                handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
                cpu, created, process_at = None, None, None
                name = entry.szExeFile
                if handle:
                    try:
                        # The Toolhelp PID can be reused before OpenProcess.
                        # Couple its name and CPU identity to the same handle.
                        path = ctypes.create_unicode_buffer(32768)
                        length = wintypes.DWORD(len(path))
                        name = (os.path.basename(path.value)
                                if kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(length))
                                else "<unavailable>")
                        values = [wintypes.FILETIME() for _ in range(4)]
                        if kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
                            process_at = time.monotonic()
                            created = _filetime(values[0])
                            cpu = (_filetime(values[2]) + _filetime(values[3])) / 10_000_000
                    finally:
                        kernel.CloseHandle(handle)
                rows[pid] = {"pid": pid, "name": name, "created": created, "cpu_seconds": cpu, "captured_at": process_at}
            present = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(snapshot)
    return {"processes": rows, "busy_seconds": busy_seconds, "captured_at": captured_at,
            "active_logical_processors": processors, "own_cpu_before_system": own_before,
            "own_cpu_after_system": own_after}


def _evaluate(before: dict, after: dict, elapsed: float, excluded_pid: int) -> dict:
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("CPU sampling interval must be positive and finite")
    wall_elapsed = elapsed
    if "captured_at" in before and "captured_at" in after:
        elapsed = after["captured_at"] - before["captured_at"]
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Invalid system CPU counter sampling interval")
    reasons, processes = [], []
    if max(before.get("active_logical_processors", 1), after.get("active_logical_processors", 1)) > 64:
        reasons.append("idle CPU proof is unavailable above 64 active logical processors; GetSystemTimes is group-scoped")
    # Only this inner interval is known to belong to the system counter window.
    # Later process enumeration can accrue sampler CPU after that window ends.
    excluded_cores = 0.0
    if "own_cpu_after_system" in before and "own_cpu_before_system" in after:
        own_seconds = after["own_cpu_before_system"] - before["own_cpu_after_system"]
        if not math.isfinite(own_seconds):
            raise RuntimeError("Invalid sampler CPU time delta")
        excluded_cores = max(0.0, own_seconds) / elapsed
    for pid, previous in before["processes"].items():
        current = after["processes"].get(pid)
        if (pid != excluded_pid and previous["name"].lower() in COMPILERS
                and (current is None or previous["created"] != current["created"])):
            reasons.append(f"compiler process {pid} ({previous['name']}) exited during the sampling interval")
    for pid, current in after["processes"].items():
        previous = before["processes"].get(pid)
        seconds = current["cpu_seconds"]
        if seconds is None:
            if current["name"].lower() in COMPILERS or current["name"] == "<unavailable>":
                reasons.append(f"compiler process {pid} ({current['name']}) CPU time is unavailable")
            continue
        process_elapsed = elapsed
        if previous and previous["created"] == current["created"]:
            if previous["cpu_seconds"] is None:
                continue
            seconds -= previous["cpu_seconds"]
            if previous.get("captured_at") is not None and current.get("captured_at") is not None:
                process_elapsed = current["captured_at"] - previous["captured_at"]
                if not math.isfinite(process_elapsed) or process_elapsed <= 0:
                    raise RuntimeError("Invalid process CPU counter sampling interval")
        # A new/reused PID's complete lifetime CPU is a conservative upper bound.
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError("Invalid process CPU time delta")
        cores = seconds / process_elapsed
        if pid == excluded_pid:
            continue
        compiler = current["name"].lower() in COMPILERS or current["name"] == "<unavailable>"
        limit = POLICY["compiler_core_limit" if compiler else "external_process_core_limit"]
        if cores > limit:
            reasons.append(f"process {pid} ({current['name']}) uses {cores:.3f} cores, limit {limit:.2f}")
        if seconds:
            processes.append({"pid": pid, "name": current["name"], "cpu_seconds": seconds, "cpu_cores": cores})
    system_seconds = after["busy_seconds"] - before["busy_seconds"]
    if not math.isfinite(system_seconds) or system_seconds < 0:
        raise RuntimeError("Invalid system CPU time delta")
    measured_cores = sum(row["cpu_cores"] for row in processes)
    system_external_cores = max(0.0, system_seconds / elapsed - excluded_cores)
    # GetSystemTimes covers the caller's primary processor group above 64 CPUs.
    # Observed process time is a second lower bound, never discarded in its favor.
    external_cores = max(system_external_cores, measured_cores)
    unattributed_cores = max(0.0, system_external_cores - measured_cores)
    if external_cores > POLICY["aggregate_external_core_limit"]:
        reasons.append(f"aggregate external CPU load {external_cores:.3f} cores exceeds limit")
    if unattributed_cores > POLICY["unattributed_core_limit"]:
        reasons.append(f"unattributed/short-lived CPU load {unattributed_cores:.3f} cores exceeds limit")
    return {"idle": not reasons, "processes": sorted(processes, key=lambda row: (-row["cpu_cores"], row["pid"])),
            "reasons": reasons, "policy": dict(POLICY), "elapsed_seconds": elapsed, "wall_elapsed_seconds": wall_elapsed,
            "external_cpu_cores": external_cores, "unattributed_cpu_cores": unattributed_cores,
            "excluded_pid": excluded_pid, "contained_sampler_cpu_cores": excluded_cores}


def sample_load(interval_seconds: float = 1) -> dict:
    if not math.isfinite(interval_seconds) or not 0.05 <= interval_seconds <= 60:
        raise ValueError("CPU sampling interval must be between 0.05 and 60 seconds")
    start = time.monotonic()
    before = _snapshot()
    time.sleep(interval_seconds)
    after = _snapshot()
    return _evaluate(before, after, time.monotonic() - start, os.getpid())
