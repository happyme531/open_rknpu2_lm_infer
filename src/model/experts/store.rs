//! Page-aligned native expert records. The checkpoint itself remains immutable.
use crate::model::checkpoint::SourceStamp;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::{Path, PathBuf},
    ptr::NonNull,
};

const PAGE: usize = 4096;
const MAGIC: &[u8; 8] = b"RKMOE01\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    pub native_version: u32,
    pub layers: usize,
    pub experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub down_bits: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Header {
    geometry: Geometry,
    sources: Vec<SourceStamp>,
    record_bytes: usize,
    record_stride: usize,
    data_offset: u64,
}

pub struct Store {
    file: File,
    header: Header,
}
impl Store {
    pub fn open(
        directory: &Path,
        geometry: &Geometry,
        sources: &[SourceStamp],
        direct: bool,
    ) -> Result<Self> {
        let path = directory.join("experts.npu");
        let mut file = File::open(&path)?;
        let mut prefix = [0u8; 16];
        file.read_exact(&mut prefix)?;
        ensure!(&prefix[..8] == MAGIC, "not a native expert cache");
        let length = u64::from_le_bytes(prefix[8..].try_into().unwrap());
        ensure!(
            length <= file.metadata()?.len().saturating_sub(16),
            "truncated expert header"
        );
        let mut json = vec![0; usize::try_from(length)?];
        file.read_exact(&mut json)?;
        let header: Header = serde_json::from_slice(&json)?;
        ensure!(
            &header.geometry == geometry && header.sources == sources && geometry.down_bits == 4,
            "expert cache does not match checkpoint/recipe"
        );
        ensure!(
            header.record_bytes > 0
                && header.record_bytes <= header.record_stride
                && header.record_stride.is_multiple_of(PAGE)
                && header.data_offset.is_multiple_of(PAGE as u64)
                && header.data_offset >= 16 + length,
            "invalid expert record layout"
        );
        let records = geometry
            .layers
            .checked_mul(geometry.experts)
            .context("expert count overflow")?;
        let bytes = (records as u64)
            .checked_mul(header.record_stride as u64)
            .and_then(|n| n.checked_add(header.data_offset))
            .context("expert cache size overflow")?;
        ensure!(file.metadata()?.len() == bytes, "incomplete expert cache");
        Self { file, header }.with_direct(direct)
    }
    pub(super) fn with_direct(self, direct: bool) -> Result<Self> {
        if direct {
            let flags = unsafe { libc::fcntl(self.file.as_raw_fd(), libc::F_GETFL) };
            ensure!(
                flags >= 0
                    && unsafe {
                        libc::fcntl(self.file.as_raw_fd(), libc::F_SETFL, flags | libc::O_DIRECT)
                    } == 0,
                "enable direct expert reads: {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(self)
    }
    pub fn geometry(&self) -> &Geometry {
        &self.header.geometry
    }
    pub fn record_bytes(&self) -> usize {
        self.header.record_bytes
    }
    pub fn record_stride(&self) -> usize {
        self.header.record_stride
    }
    pub fn read(&self, layer: usize, expert: usize) -> Result<Record> {
        ensure!(
            layer < self.header.geometry.layers && expert < self.header.geometry.experts,
            "expert file index out of range"
        );
        let mut data = Record::new(self.header.record_stride, self.header.record_bytes)?;
        let offset = self.header.data_offset
            + ((layer * self.header.geometry.experts + expert) * self.header.record_stride) as u64;
        self.file
            .read_exact_at(data.all_mut(), offset)
            .with_context(|| format!("read expert {layer}/{expert}"))?;
        Ok(data)
    }

    /// Strata's `copy_blobs` batches neighbouring expert records. `preadv`
    /// writes into the existing aligned slots, without an extra staging copy.
    pub fn read_batch(&self, layer: usize, first: usize, count: usize) -> Result<Vec<Record>> {
        ensure!(
            count > 0
                && layer < self.header.geometry.layers
                && first + count <= self.header.geometry.experts,
            "expert batch out of range"
        );
        let mut records = (0..count)
            .map(|_| Record::new(self.header.record_stride, self.header.record_bytes))
            .collect::<Result<Vec<_>>>()?;
        let offset = self.header.data_offset
            + ((layer * self.header.geometry.experts + first) * self.header.record_stride) as u64;
        let total = count * self.header.record_stride;
        let mut done = 0;
        while done < total {
            let mut skip = done;
            let vectors: Vec<_> = records
                .iter_mut()
                .filter_map(|record| {
                    let bytes = record.all_mut();
                    if skip >= bytes.len() {
                        skip -= bytes.len();
                        return None;
                    }
                    let bytes = &mut bytes[skip..];
                    skip = 0;
                    Some(libc::iovec {
                        iov_base: bytes.as_mut_ptr().cast(),
                        iov_len: bytes.len(),
                    })
                })
                .collect();
            let got = unsafe {
                libc::preadv(
                    self.file.as_raw_fd(),
                    vectors.as_ptr(),
                    vectors.len() as i32,
                    (offset + done as u64) as libc::off_t,
                )
            };
            if got < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error.into());
            }
            ensure!(got > 0, "truncated expert batch");
            done += got as usize;
        }
        Ok(records)
    }
}

/// Single owned read buffer, suitable for O_DIRECT and transfer to an I/O worker.
pub struct Record {
    pointer: NonNull<u8>,
    length: usize,
    raw: usize,
}
// The allocation has a single owner. No references survive a move to another thread.
unsafe impl Send for Record {}
impl Record {
    fn new(length: usize, raw: usize) -> Result<Self> {
        let layout = Layout::from_size_align(length, PAGE)?;
        let pointer =
            NonNull::new(unsafe { alloc_zeroed(layout) }).context("allocate expert read buffer")?;
        Ok(Self {
            pointer,
            length,
            raw,
        })
    }
    fn all_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.raw) }
    }
}
impl Drop for Record {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.pointer.as_ptr(),
                Layout::from_size_align(self.length, PAGE).unwrap(),
            )
        };
    }
}

pub fn lock(directory: &Path) -> Result<File> {
    std::fs::create_dir_all(directory)?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("prepare.lock"))?;
    ensure!(
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0,
        "lock expert cache: {}",
        std::io::Error::last_os_error()
    );
    Ok(file)
}

pub struct Writer {
    file: File,
    path: PathBuf,
    final_path: PathBuf,
    header: Header,
    written: usize,
    finished: bool,
}
impl Writer {
    /// Hold `lock(directory)` while constructing and publishing a cache.
    pub fn create(
        directory: &Path,
        geometry: Geometry,
        sources: Vec<SourceStamp>,
        record_bytes: usize,
    ) -> Result<Self> {
        ensure!(record_bytes > 0, "empty expert record");
        std::fs::create_dir_all(directory)?;
        let mut header = Header {
            geometry,
            sources,
            record_bytes,
            record_stride: record_bytes.div_ceil(PAGE) * PAGE,
            data_offset: PAGE as u64,
        };
        let json = loop {
            let json = serde_json::to_vec(&header)?;
            let offset = (16 + json.len()).div_ceil(PAGE) * PAGE;
            if header.data_offset == offset as u64 {
                break json;
            }
            header.data_offset = offset as u64;
        };
        let path = directory.join(format!("experts.npu.tmp.{}", std::process::id()));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(json.len() as u64).to_le_bytes())?;
        file.write_all(&json)?;
        file.write_all(&vec![0; header.data_offset as usize - 16 - json.len()])?;
        Ok(Self {
            file,
            path,
            final_path: directory.join("experts.npu"),
            header,
            written: 0,
            finished: false,
        })
    }
    pub fn write(&mut self, layer: usize, expert: usize, data: &[u8]) -> Result<()> {
        ensure!(
            self.written == layer * self.header.geometry.experts + expert
                && layer < self.header.geometry.layers
                && expert < self.header.geometry.experts
                && data.len() == self.header.record_bytes,
            "expert export order/size mismatch"
        );
        self.file.write_all(data)?;
        self.file
            .write_all(&vec![0; self.header.record_stride - data.len()])?;
        self.written += 1;
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        ensure!(
            self.written == self.header.geometry.layers * self.header.geometry.experts,
            "missing exported experts"
        );
        self.file.sync_all()?;
        unsafe { libc::posix_fadvise(self.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        std::fs::rename(&self.path, &self.final_path)?;
        self.finished = true;
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        if !self.finished {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positional_records_are_page_aligned_and_partial_files_are_rejected() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let geometry = Geometry {
            native_version: 1,
            layers: 1,
            experts: 2,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let mut writer = Writer::create(dir.path(), geometry.clone(), vec![], 7936)?;
        writer.write(0, 0, &vec![17; 7936])?;
        writer.write(0, 1, &vec![29; 7936])?;
        writer.finish()?;
        let store = Store::open(dir.path(), &geometry, &[], false)?;
        assert_eq!(store.read(0, 1)?.as_bytes(), vec![29; 7936]);
        assert_eq!(store.read(0, 0)?.as_bytes(), vec![17; 7936]);
        let batch = store.read_batch(0, 0, 2)?;
        assert_eq!(batch[0].as_bytes(), vec![17; 7936]);
        assert_eq!(batch[1].as_bytes(), vec![29; 7936]);
        assert!(store.read(1, 0).is_err());
        OpenOptions::new()
            .write(true)
            .open(dir.path().join("experts.npu"))?
            .set_len(4096)?;
        assert!(Store::open(dir.path(), &geometry, &[], false).is_err());
        Ok(())
    }
}
