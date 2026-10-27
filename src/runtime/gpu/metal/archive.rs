//! Bundled native pipeline archive; extraction is private and process-local.
use super::*;
use objc2_foundation::{NSArray, NSURL};
use std::{
    fs::{self, DirBuilder, OpenOptions},
    io::Write,
    os::unix::fs::DirBuilderExt,
    path::PathBuf,
};

struct Extracted {
    directory: PathBuf,
}
impl Drop for Extracted {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.directory.join("ocr.metallib"));
        let _ = fs::remove_dir(&self.directory);
    }
}
pub(super) struct Archive {
    raw: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    _file: Extracted,
}
// SAFETY: the archive is fully initialized before publication and thereafter
// only read by Metal pipeline creation. No archive mutation API is exposed.
unsafe impl Send for Archive {}
unsafe impl Sync for Archive {}

impl Archive {
    pub(super) fn load(device: &Device) -> Option<Self> {
        if device.raw.architecture().name().to_string() != "applegpu_g16g" {
            return None;
        }
        if objc2_foundation::NSProcessInfo::processInfo()
            .operatingSystemVersion()
            .majorVersion
            != env!("SIMD_OCR_METAL_ARCHIVE_OS").parse::<isize>().ok()?
        {
            return None;
        }
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let mut file = None;
        for _ in 0..16 {
            let directory = std::env::temp_dir().join(format!(
                "simd-ocr-metal-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match DirBuilder::new().mode(0o700).create(&directory) {
                Ok(()) => {
                    file = Some(Extracted { directory });
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(_) => return None,
            }
        }
        let file = file?;
        let path = file.directory.join("ocr.metallib");
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .ok()?;
        let bytes = embedded_resource(
            include_bytes!(concat!(env!("OUT_DIR"), "/ocr.binary.metallib.deflate")),
            env!("SIMD_OCR_METAL_BINARY_BYTES"),
        )
        .ok()?;
        output.write_all(&bytes).ok()?;
        drop(output);
        let descriptor = MTLBinaryArchiveDescriptor::new();
        descriptor.setUrl(Some(&NSURL::fileURLWithPath(&NSString::from_str(
            path.to_str()?,
        ))));
        let raw = device
            .raw
            .newBinaryArchiveWithDescriptor_error(&descriptor)
            .map_err(|e| {
                #[cfg(test)]
                eprintln!("native archive load: {e}");
                let _ = e;
            })
            .ok()?;
        Some(Self { raw, _file: file })
    }
    pub(super) fn pipeline(
        &self,
        device: &Device,
        function: &ProtocolObject<dyn MTLFunction>,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(function));
        descriptor.setBinaryArchives(Some(&NSArray::from_slice(&[&*self.raw])));
        device
            .raw
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::FailOnBinaryArchiveMiss,
                None,
            )
            .map_err(|e| {
                #[cfg(test)]
                eprintln!("native archive pipeline: {e}");
                let _ = e;
            })
            .ok()
    }
}
