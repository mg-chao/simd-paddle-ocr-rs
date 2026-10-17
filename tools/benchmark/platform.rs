//! Native process counters and explicit scheduling policy for each supported host.
use super::{Result, UNBOUND_AFFINITY};
use std::time::Duration;

pub fn affinity(threads: usize, explicit: Option<u64>) -> Result<String> {
    #[cfg(windows)]
    {
        let available = super::windows::allowed_affinity()?;
        let mask = match explicit {
            Some(mask) => mask,
            None => super::select_affinity(available, threads)?,
        };
        if mask.count_ones() != threads as u32 || mask & !available != 0 {
            return Err("affinity must match the thread budget and parent process mask".into());
        }
        return Ok(format!("{mask:x}"));
    }
    #[cfg(target_os = "macos")]
    {
        if explicit.is_some() {
            return Err(
                "--affinity is unsupported on macOS; workers use unpinned scheduling".into(),
            );
        }
        if threads > std::thread::available_parallelism()?.get() {
            return Err("thread budget exceeds the host's available parallelism".into());
        }
        Ok(UNBOUND_AFFINITY.into())
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = (threads, explicit);
        Err("benchmark requires Windows x64 or 64-bit macOS".into())
    }
}

pub fn configure(affinity: &str) -> Result<()> {
    #[cfg(windows)]
    {
        return super::windows::set_affinity(u64::from_str_radix(affinity, 16)?);
    }
    #[cfg(target_os = "macos")]
    {
        if affinity != UNBOUND_AFFINITY {
            return Err("macOS benchmark jobs must declare unbound scheduling".into());
        }
        Ok(())
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = affinity;
        Err("unsupported benchmark host".into())
    }
}

pub fn process_cpu_time() -> Result<Duration> {
    #[cfg(windows)]
    {
        return super::windows::process_cpu_time();
    }
    #[cfg(target_os = "macos")]
    {
        let usage = usage()?;
        let time = |value: libc::timeval| -> Result<Duration> {
            if !(0..1_000_000).contains(&value.tv_usec) {
                return Err("invalid process CPU microseconds".into());
            }
            Ok(Duration::new(
                u64::try_from(value.tv_sec)?,
                value.tv_usec as u32 * 1000,
            ))
        };
        time(usage.ru_utime)?
            .checked_add(time(usage.ru_stime)?)
            .ok_or_else(|| "process CPU time overflow".into())
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        Err("unsupported benchmark host".into())
    }
}

/// (Windows private commit, optional macOS resident bytes), never conflated.
pub fn peak_memory() -> Result<(u64, Option<u64>)> {
    #[cfg(windows)]
    {
        return Ok((super::windows::peak_private()?, None));
    }
    #[cfg(target_os = "macos")]
    {
        Ok((0, Some(u64::try_from(usage()?.ru_maxrss)?)))
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        Err("unsupported benchmark host".into())
    }
}

#[cfg(target_os = "macos")]
fn usage() -> Result<libc::rusage> {
    let mut usage = std::mem::MaybeUninit::uninit();
    // SAFETY: the SDK-compatible libc rusage buffer is fully written on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(unsafe { usage.assume_init() })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    #[test]
    fn darwin_counters_and_unpinned_policy_are_valid() {
        assert_eq!(std::mem::size_of::<libc::rusage>(), 144);
        assert_eq!(affinity(1, None).unwrap(), UNBOUND_AFFINITY);
        assert!(affinity(1, Some(1)).is_err());
        assert!(configure("1").is_err());
        configure(UNBOUND_AFFINITY).unwrap();
        let before = process_cpu_time().unwrap();
        let (private, resident) = peak_memory().unwrap();
        assert_eq!(private, 0);
        assert!(resident.unwrap() > 0);
        assert!(process_cpu_time().unwrap() >= before);
    }
}
