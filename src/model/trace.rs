use anyhow::{Context, Result};
use std::{io::Write, path::PathBuf};

pub(crate) struct Trace {
    directory: Option<PathBuf>,
    #[cfg(test)]
    position: Option<usize>,
}

impl Trace {
    #[cfg(test)]
    pub fn in_directory(path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&path)?;
        Ok(Self {
            directory: Some(path),
            position: None,
        })
    }
    #[cfg(test)]
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
