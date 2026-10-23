use anyhow::{Context, Result};
use std::ffi::CStr;
use std::{io::Write, path::PathBuf};

#[cfg(rknpu_trace)]
unsafe extern "C" {
    fn rknpu2_trace_start(path: *const std::ffi::c_char) -> i32;
    fn rknpu2_trace_stop() -> i32;
    fn rknpu2_trace_begin(name: *const std::ffi::c_char, layer: i32, rows: i32, position: i32);
    fn rknpu2_trace_end();
}

/// Perfetto span in the NPU library's existing `llm` category.
pub(crate) struct Span;
impl Span {
    #[inline]
    pub fn new(name: &'static CStr, layer: i32, rows: usize, position: usize) -> Self {
        #[cfg(rknpu_trace)]
        unsafe {
            rknpu2_trace_begin(name.as_ptr(), layer, rows as i32, position as i32);
        }
        #[cfg(not(rknpu_trace))]
        let _ = (name, layer, rows, position);
        Self
    }
}
impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        #[cfg(rknpu_trace)]
        unsafe {
            rknpu2_trace_end();
        }
    }
}
#[inline]
#[cfg(feature = "model-ling3tiny")]
pub(crate) fn measure<T>(name: &'static CStr, operation: impl FnOnce() -> T) -> T {
    let _span = Span::new(name, -1, 0, 0);
    operation()
}

/// One native Perfetto recording per measured benchmark repetition.
pub(crate) struct Recording {
    active: bool,
}
impl Recording {
    pub fn for_sample(case: usize, repetition: usize) -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("OPEN_RKNPU2_PROFILE_TRACE") else {
            return Ok(None);
        };
        #[cfg(not(rknpu_trace))]
        {
            let _ = (path, case, repetition);
            anyhow::bail!("Perfetto requires RK_NPU_ENABLE_TRACE=ON at build time");
        }
        #[cfg(rknpu_trace)]
        {
            use std::{ffi::CString, os::unix::ffi::OsStrExt};
            let path =
                PathBuf::from(path).with_extension(format!("case{case}.rep{repetition}.pftrace"));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let path = CString::new(path.as_os_str().as_bytes())?;
            anyhow::ensure!(
                unsafe { rknpu2_trace_start(path.as_ptr()) } == 0,
                "start Perfetto recording failed"
            );
            Ok(Some(Self { active: true }))
        }
    }
    pub fn finish(mut self) -> Result<()> {
        self.active = false;
        #[cfg(rknpu_trace)]
        anyhow::ensure!(
            unsafe { rknpu2_trace_stop() } == 0,
            "finish Perfetto recording failed"
        );
        Ok(())
    }
}
impl Drop for Recording {
    fn drop(&mut self) {
        if self.active {
            #[cfg(rknpu_trace)]
            unsafe {
                rknpu2_trace_stop();
            }
        }
    }
}

pub(crate) struct Trace {
    directory: Option<PathBuf>,
    #[cfg(test)]
    position: Option<usize>,
}

impl Trace {
    #[cfg(all(test, feature = "arch-qwen35"))]
    pub fn in_directory(path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&path)?;
        Ok(Self {
            directory: Some(path),
            position: None,
        })
    }
    #[cfg(all(test, feature = "model-minicpmv47"))]
    pub fn at_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }
    pub fn from_environment() -> Result<Self> {
        let directory = std::env::var_os("OPEN_RKNPU2_TRACE_DIR")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        if let Some(path) = &directory {
            std::fs::create_dir_all(path)?;
        }
        Ok(Self {
            directory,
            #[cfg(test)]
            position: None,
        })
    }
    #[cfg(any(feature = "arch-dense", feature = "arch-qwen35", feature = "vision"))]
    pub fn enabled(&self) -> bool {
        self.directory.is_some()
    }
    pub fn path(&self) -> Option<&std::path::Path> {
        self.directory.as_deref()
    }
    pub fn save(&self, position: usize, layer: i32, stage: &str, values: &[f32]) -> Result<()> {
        #[cfg(test)]
        if self.position.is_some_and(|p| p != position) {
            return Ok(());
        }
        let Some(directory) = &self.directory else {
            return Ok(());
        };
        let path = directory.join(format!("p{position}-l{layer}-{stage}.f32"));
        let mut file = std::fs::File::create(&path)
            .with_context(|| format!("write trace {}", path.display()))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        file.write_all(bytes)?;
        Ok(())
    }
    #[cfg(feature = "vision")]
    pub fn save_half(
        &self,
        position: usize,
        layer: i32,
        stage: &str,
        values: &[u16],
    ) -> Result<()> {
        if self.enabled() {
            self.save(
                position,
                layer,
                stage,
                &crate::ops::vision_half::widen(values),
            )?;
        }
        Ok(())
    }
}
