use crate::npu::QuantMatrix;
use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions, UncheckedAdvice};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    ops::Range,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

#[derive(Clone)]
pub struct Tensor {
    map: Arc<Mmap>,
    range: Range<usize>,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn at(&self, index: usize) -> Result<Self> {
        ensure!(
            !self.shape.is_empty() && index < self.shape[0],
            "tensor slice out of range"
        );
        let bytes = self.range.len() / self.shape[0];
        let start = self.range.start + index * bytes;
        Ok(Self {
            map: self.map.clone(),
            range: start..start + bytes,
            dtype: self.dtype,
            shape: self.shape[1..].to_vec(),
        })
    }
    pub fn data(&self) -> &[u8] {
        &self.map[self.range.clone()]
    }
    pub fn require_shape(&self, shape: &[usize], name: &str) -> Result<()> {
        ensure!(
            self.shape == shape,
            "unexpected shape: {name}: {:?}, expected {shape:?}",
            self.shape
        );
        Ok(())
    }
    pub fn floating(&self) -> bool {
        matches!(self.dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16)
    }
    pub fn number(&self, i: usize) -> f32 {
        let b = self.data();
        match self.dtype {
            Dtype::F32 => f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()),
            Dtype::F16 => {
                f16::from_bits(u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap())).to_f32()
            }
            Dtype::BF16 => {
                bf16::from_bits(u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap()))
                    .to_f32()
            }
            _ => unreachable!("validated floating tensor"),
        }
    }
    pub fn integer(&self, i: usize) -> Result<i64> {
        let b = self.data();
        Ok(match self.dtype {
            Dtype::I8 => b[i] as i8 as i64,
            Dtype::I32 => i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as i64,
            Dtype::I64 => i64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()),
            _ => bail!("expected integer tensor"),
        })
    }
    pub fn floats(&self) -> Result<Vec<f32>> {
        ensure!(self.floating(), "expected floating tensor");
        let values: Vec<_> = (0..self.shape.iter().product())
            .map(|i| self.number(i))
            .collect();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite model tensor"
        );
        Ok(values)
    }
    pub fn row(&self, row: usize, output: &mut [f32]) {
        let width = self.shape[1];
        let mut i = 0;
        #[cfg(target_arch = "aarch64")]
        if self.dtype == Dtype::BF16 {
            unsafe {
                use std::arch::aarch64::*;
                let pointer = self.data().as_ptr().cast::<u16>().add(row * width);
                while i + 8 <= output.len() {
                    let values = vld1q_u16(pointer.add(i));
                    vst1q_f32(
                        output.as_mut_ptr().add(i),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(values))),
                    );
                    vst1q_f32(
                        output.as_mut_ptr().add(i + 4),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(values))),
                    );
                    i += 8;
                }
            }
        }
        while i < output.len() {
            output[i] = self.number(row * width + i);
            i += 1;
        }
    }
}

pub struct TensorStore {
    tensors: HashMap<String, Tensor>,
    maps: Vec<Arc<Mmap>>,
    files: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceStamp {
    name: String,
    bytes: u64,
    modified_ns: u64,
}

impl TensorStore {
    pub fn open(directory: &Path) -> Result<Self> {
        let mut store = Self {
            tensors: HashMap::new(),
            maps: Vec::new(),
            files: Vec::new(),
        };
        let index_path = directory.join("model.safetensors.index.json");
        if index_path.exists() {
            let index: Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
            let weights = index["weight_map"]
                .as_object()
                .context("missing safetensors weight_map")?;
            let mut shards = BTreeSet::new();
            for value in weights.values() {
                let name = value.as_str().context("invalid shard filename")?;
                let mut components = Path::new(name).components();
                ensure!(
                    matches!(components.next(), Some(Component::Normal(_)))
                        && components.next().is_none(),
                    "shard index requires plain filenames"
                );
                shards.insert(name.to_owned());
            }
            store.files.push(index_path);
            for shard in shards {
                store.add_file(&directory.join(&shard), Some(weights))?;
            }
            ensure!(
                store.tensors.len() == weights.len(),
                "index has missing tensors"
            );
        } else {
            store.add_file(&directory.join("model.safetensors"), None)?;
        }
        Ok(store)
    }
    fn add_file(
        &mut self,
        path: &Path,
        index: Option<&serde_json::Map<String, Value>>,
    ) -> Result<()> {
        let file =
            File::open(path).with_context(|| format!("cannot open shard {}", path.display()))?;
        ensure!(file.metadata()?.is_file(), "invalid safetensors shard");
        // Checkpoint files stay immutable while the model is alive.
        let map = Arc::new(unsafe { MmapOptions::new().map(&file)? });
        let tensors = SafeTensors::deserialize(&map)
            .with_context(|| format!("invalid safetensors shard {}", path.display()))?;
        for (name, view) in tensors.tensors() {
            if let Some(index) = index {
                ensure!(
                    index.get(&name).and_then(Value::as_str)
                        == path.file_name().and_then(|p| p.to_str()),
                    "tensor/index mismatch: {name}"
                );
            }
            ensure!(
                matches!(
                    view.dtype(),
                    Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::I8 | Dtype::I32 | Dtype::I64
                ),
                "unsupported safetensors dtype: {:?}",
                view.dtype()
            );
            let start = view.data().as_ptr() as usize - map.as_ptr() as usize;
            let tensor = Tensor {
                map: map.clone(),
                range: start..start + view.data().len(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
            };
            ensure!(
                self.tensors.insert(name.clone(), tensor).is_none(),
                "duplicate tensor: {name}"
            );
        }
        drop(tensors);
        self.maps.push(map);
        self.files.push(path.to_owned());
        Ok(())
    }
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .with_context(|| format!("missing tensor: {name}"))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn mapped_bytes(&self) -> usize {
        self.maps.iter().map(|m| m.len()).sum()
    }
    pub fn release_pages(&self) -> Result<()> {
        for map in &self.maps {
            unsafe {
                map.unchecked_advise(UncheckedAdvice::DontNeed)?;
            }
        }
        Ok(())
    }
}

pub struct QuantLoader {
    pub store: TensorStore,
    quant: Value,
    c4: Option<HashMap<String, u32>>,
    metadata_files: Vec<PathBuf>,
}

fn matches(target: &str, module: &str) -> Result<bool> {
    if target == "Linear" || target == module {
        return Ok(true);
    }
    if let Some(pattern) = target.strip_prefix("re:") {
        return Ok(regex::Regex::new(&format!("^(?:{pattern})$"))?.is_match(module));
    }
    Ok(false)
}
fn symmetric(q: &Value, bits: u32, strategy: &str, dynamic: bool) -> Result<()> {
    ensure!(
        q["num_bits"].as_u64() == Some(bits as u64)
            && q["type"].as_str() == Some("int")
            && q["strategy"].as_str() == Some(strategy)
            && q["symmetric"].as_bool() == Some(true)
            && q.get("dynamic")
                .map_or(!dynamic, |v| v.as_bool() == Some(dynamic))
            && q["group_size"].is_null()
            && q["actorder"].is_null(),
        "expected symmetric static channel W4/W8 and dynamic token A8; unsupported group scales/actorder"
    );
    Ok(())
}

fn transpose_codes(codes: &mut [i8], n: usize, read: impl Fn(usize, usize) -> i8 + Sync) {
    codes
        .par_chunks_mut(n * 32)
        .enumerate()
        .for_each(|(block_index, block)| {
            for n0 in (0..n).step_by(64) {
                for (local_k, row) in block.chunks_mut(n).enumerate() {
                    let k = block_index * 32 + local_k;
                    for (ni, value) in row.iter_mut().enumerate().take((n0 + 64).min(n)).skip(n0) {
                        *value = read(k, ni);
                    }
                }
            }
        });
}

impl QuantLoader {
    pub fn open(directory: &Path, config: &Value) -> Result<Self> {
        let store = TensorStore::open(directory)?;
        if config["c4_quantization"]["format"] == "minicpmv47-c4-packed-v1" {
            let manifest: Value =
                serde_json::from_slice(&std::fs::read(directory.join("bitwidth_manifest.json"))?)?;
            ensure!(
                manifest["format"] == "minicpmv47-c4-packed-v1",
                "C4 manifest format mismatch"
            );
            let bits: HashMap<String, u32> =
                serde_json::from_value(manifest["linear_bits"].clone())?;
            ensure!(
                bits.values().all(|b| matches!(b, 4 | 8)),
                "unsupported C4 weight precision"
            );
            for (name, tensor) in &store.tensors {
                if name.ends_with(".input_scale") {
                    ensure!(
                        tensor.floats()?.iter().all(|s| *s == 1.),
                        "non-unit input_scale is unsupported: {name}"
                    );
                }
            }
            return Ok(Self {
                store,
                quant: Value::Null,
                c4: Some(bits),
                metadata_files: vec![
                    directory.join("config.json"),
                    directory.join("bitwidth_manifest.json"),
                ],
            });
        }
        let mut loader = Self::new(store, config)?;
        loader.metadata_files.push(directory.join("config.json"));
        Ok(loader)
    }
    /// Metadata for files consumed by this loader, without reopening its index.
    pub fn source_stamps(&self) -> Result<Vec<SourceStamp>> {
        let mut stamps = self
            .store
            .files
            .iter()
            .chain(&self.metadata_files)
            .map(|path| {
                let meta = std::fs::metadata(path)?;
                Ok(SourceStamp {
                    name: path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .context("invalid checkpoint filename")?
                        .to_owned(),
                    bytes: meta.len(),
                    modified_ns: meta
                        .modified()?
                        .duration_since(UNIX_EPOCH)?
                        .as_nanos()
                        .try_into()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        stamps.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(stamps)
    }
    pub fn is_c4(&self) -> bool {
        self.c4.is_some()
    }
    pub fn quantized(&self, module: &str) -> bool {
        self.c4.as_ref().map_or_else(
            || {
                self.store.contains(&format!("{module}.weight_packed"))
                    || self
                        .store
                        .get(&format!("{module}.weight"))
                        .is_ok_and(|tensor| !tensor.floating())
            },
            |bits| bits.contains_key(module),
        )
    }
    pub fn expert(
        &self,
        module: &str,
        expert: usize,
        h: usize,
        f: usize,
    ) -> Result<(QuantMatrix, QuantMatrix)> {
        if !self.is_c4() {
            let name = format!("{module}.{expert}");
            return Ok((
                concatenate(vec![
                    self.load(&format!("{name}.gate_proj"), f, h)?,
                    self.load(&format!("{name}.up_proj"), f, h)?,
                ])?,
                self.load(&format!("{name}.down_proj"), h, f)?,
            ));
        }
        let matrix = |name: &str, n, k| -> Result<QuantMatrix> {
            let packed = self
                .store
                .get(&format!("{module}.{name}_packed"))?
                .at(expert)?;
            let scale = self
                .store
                .get(&format!("{module}.{name}_scale"))?
                .at(expert)?;
            unpack_matrix(&packed, &scale, 4, n, k)
        };
        Ok((
            matrix("gate_up_proj", 2 * f, h)?,
            matrix("down_proj", h, f)?,
        ))
    }
    pub fn new(store: TensorStore, config: &Value) -> Result<Self> {
        let quant = config["quantization_config"].clone();
        ensure!(
            quant["quant_method"] == "compressed-tensors"
                && quant["quantization_status"] == "compressed",
            "expected a compressed-tensors compressed checkpoint"
        );
        for key in ["sparsity_config", "transform_config"] {
            let v = &quant[key];
            ensure!(
                v.is_null() || v.as_object().is_some_and(|v| v.is_empty()),
                "unsupported compressed-tensors {key}"
            );
        }
        ensure!(
            quant["kv_cache_scheme"].is_null(),
            "quantized KV cache schemes are unsupported"
        );
        ensure!(
            quant["config_groups"].is_object(),
            "missing quantization config_groups"
        );
        Ok(Self {
            store,
            quant,
            c4: None,
            metadata_files: Vec::new(),
        })
    }
    fn scheme(&self, module: &str) -> Result<&Value> {
        if let Some(ignore) = self.quant["ignore"].as_array() {
            for target in ignore {
                ensure!(
                    !matches(target.as_str().context("invalid ignore target")?, module)?,
                    "required projection excluded from quantization: {module}"
                );
            }
        }
        let mut found = None;
        for value in self.quant["config_groups"].as_object().unwrap().values() {
            for target in value["targets"]
                .as_array()
                .context("missing quantization targets")?
            {
                if matches(
                    target.as_str().context("invalid quantization target")?,
                    module,
                )? {
                    if let Some(previous) = found {
                        ensure!(previous == value, "ambiguous quantization scheme: {module}");
                    }
                    found = Some(value);
                }
            }
        }
        found.with_context(|| format!("no quantization scheme for {module}"))
    }
    pub fn bits(&self, module: &str) -> Result<u32> {
        if let Some(bits) = &self.c4 {
            return bits
                .get(module)
                .copied()
                .with_context(|| format!("unquantized C4 module: {module}"));
        }
        let s = self.scheme(module)?;
        let bits = s["weights"]["num_bits"]
            .as_u64()
            .context("missing weight bits")? as u32;
        ensure!(bits == 4 || bits == 8, "only W4A8 and W8A8 are supported");
        symmetric(&s["weights"], bits, "channel", false)?;
        symmetric(&s["input_activations"], 8, "token", true)?;
        ensure!(
            s["output_activations"].is_null(),
            "quantized linear output is unsupported: {module}"
        );
        Ok(bits)
    }
    pub fn load(&self, module: &str, n: usize, k: usize) -> Result<QuantMatrix> {
        ensure!(n > 0 && k > 0, "invalid linear dimensions");
        let bits = self.bits(module)?;
        ensure!(
            !self.store.contains(&format!("{module}.bias")),
            "linear bias is unsupported: {module}"
        );
        ensure!(
            !self.store.contains(&format!("{module}.weight_g_idx")),
            "weight_g_idx/actorder unsupported: {module}"
        );
        let scale = self.store.get(&format!("{module}.weight_scale"))?;
        scale.require_shape(&[n, 1], "weight_scale")?;
        let scales = scale.floats()?;
        ensure!(
            scales.iter().all(|v| *v > 0.0),
            "nonpositive scale: {module}"
        );
        if self.store.contains(&format!("{module}.weight_zero_point")) {
            let zero = self.store.get(&format!("{module}.weight_zero_point"))?;
            for i in 0..zero.shape.iter().product() {
                ensure!(zero.integer(i)? == 0, "nonzero weight zero point: {module}");
            }
        }
        let format = if self.is_c4() {
            "pack-quantized"
        } else {
            self.scheme(module)?
                .get("format")
                .or_else(|| self.quant.get("format"))
                .and_then(Value::as_str)
                .unwrap_or("")
        };
        let mut codes = vec![0i8; k.checked_mul(n).context("weight size overflow")?];
        match format {
            "pack-quantized" => {
                let packed = self.store.get(&format!("{module}.weight_packed"))?;
                let shape = self.store.get(&format!("{module}.weight_shape"))?;
                shape.require_shape(&[2], "weight_shape")?;
                ensure!(
                    shape.integer(0)? == n as i64 && shape.integer(1)? == k as i64,
                    "weight_shape mismatch: {module}"
                );
                unpack_codes(&packed, bits, n, k, &mut codes)?;
            }
            "int-quantized" if bits == 8 => {
                let weights = self.store.get(&format!("{module}.weight"))?;
                weights.require_shape(&[n, k], "weight")?;
                ensure!(weights.dtype == Dtype::I8, "W8 weight must be signed I8");
                let data = weights.data();
                transpose_codes(&mut codes, n, |ki, ni| data[ni * k + ki] as i8);
            }
            _ => bail!("unsupported compressed-tensors storage: {format}"),
        }
        self.store.release_pages()?;
        Ok(QuantMatrix {
            bits,
            k,
            n,
            codes,
            scales,
        })
    }
}

fn unpack_matrix(
    packed: &Tensor,
    scale: &Tensor,
    bits: u32,
    n: usize,
    k: usize,
) -> Result<QuantMatrix> {
    scale.require_shape(&[n, 1], "weight scale")?;
    let scales = scale.floats()?;
    ensure!(scales.iter().all(|s| *s > 0.), "nonpositive weight scale");
    let mut codes = vec![0; k * n];
    unpack_codes(packed, bits, n, k, &mut codes)?;
    Ok(QuantMatrix {
        bits,
        k,
        n,
        codes,
        scales,
    })
}

fn unpack_codes(packed: &Tensor, bits: u32, n: usize, k: usize, codes: &mut [i8]) -> Result<()> {
    let factor = 32 / bits as usize;
    let columns = k.div_ceil(factor);
    packed.require_shape(&[n, columns], "packed weight")?;
    ensure!(packed.dtype == Dtype::I32, "packed weight must be I32");
    let data = packed.data();
    transpose_codes(codes, n, |ki, ni| {
        let offset = (ni * columns + ki / factor) * 4;
        let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        (((word >> ((ki % factor) * bits as usize)) & ((1 << bits) - 1)) as i32
            - (1i32 << (bits - 1))) as i8
    });
    Ok(())
}

pub fn concatenate(mut matrices: Vec<QuantMatrix>) -> Result<QuantMatrix> {
    ensure!(!matrices.is_empty(), "empty fused projection");
    if matrices.len() == 1 {
        return Ok(matrices.pop().unwrap());
    }
    let bits = matrices[0].bits;
    let k = matrices[0].k;
    ensure!(
        matrices.iter().all(|m| m.bits == bits && m.k == k),
        "fused projections require identical bit widths and K"
    );
    let n = matrices.iter().map(|m| m.n).sum();
    let scales = matrices
        .iter()
        .flat_map(|m| m.scales.iter().copied())
        .collect();
    let mut codes = vec![0; k * n];
    codes.par_chunks_mut(n).enumerate().for_each(|(ki, row)| {
        let mut offset = 0;
        for m in &matrices {
            row[offset..offset + m.n].copy_from_slice(&m.codes[ki * m.n..(ki + 1) * m.n]);
            offset += m.n;
        }
    });
    Ok(QuantMatrix {
        bits,
        k,
        n,
        codes,
        scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_stamps_follow_loaded_files_and_invalidate_changed_sources() -> Result<()> {
        use crate::model::experts::store::{Geometry, Store, Writer};

        let dir = tempfile::tempdir()?;
        let config = serde_json::json!({"quantization_config": {
            "quant_method":"compressed-tensors", "quantization_status":"compressed",
            "config_groups":{}
        }});
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&config)?)?;
        let mut header = serde_json::to_vec(&serde_json::json!({
            "a": {"dtype":"F32", "shape":[1], "data_offsets":[0,4]},
            "b": {"dtype":"F32", "shape":[1], "data_offsets":[4,8]}
        }))?;
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend([1f32, 2.].into_iter().flat_map(f32::to_le_bytes));
        std::fs::write(dir.path().join("model.safetensors"), &file)?;
        let loader = QuantLoader::open(dir.path(), &config)?;
        assert!(loader.store.contains("a") && loader.store.contains("b"));
        let single = loader.source_stamps()?;
        assert_eq!(
            single.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["config.json", "model.safetensors"]
        );
        let geometry = Geometry {
            native_version: 1,
            layers: 1,
            experts: 1,
            hidden: 64,
            intermediate: 64,
            down_bits: 4,
        };
        let cache = dir.path().join("cache");
        let mut writer = Writer::create(&cache, geometry.clone(), single.clone(), 4)?;
        writer.write(0, 0, &[1, 2, 3, 4])?;
        writer.finish()?;
        assert!(Store::open(&cache, &geometry, &single, false).is_ok());
        std::fs::write(dir.path().join("config.json"), b"{\"changed\":true}")?;
        assert!(Store::open(&cache, &geometry, &loader.source_stamps()?, false).is_err());

        std::fs::write(dir.path().join("part.safetensors"), file)?;
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"part.safetensors","b":"part.safetensors"}}"#,
        )?;
        // An unrelated manifest is not a dependency of compressed-tensors.
        std::fs::write(dir.path().join("bitwidth_manifest.json"), b"{}")?;
        let indexed = QuantLoader::open(dir.path(), &config)?;
        assert!(indexed.store.contains("a") && indexed.store.contains("b"));
        let stamps = indexed.source_stamps()?;
        assert_eq!(
            stamps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "config.json",
                "model.safetensors.index.json",
                "part.safetensors"
            ]
        );
        std::fs::write(dir.path().join("model.safetensors.index.json"), b"changed")?;
        // Snapshotting checks metadata of the opened files, without parsing again.
        assert_ne!(indexed.source_stamps()?, stamps);
        Ok(())
    }

    fn c4_fixture() -> Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        let mut tensors = serde_json::Map::new();
        let mut data = Vec::new();
        let mut add = |name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>| {
            let start = data.len();
            data.extend(bytes);
            tensors.insert(
                name.into(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
            );
        };
        add(
            "q.weight_packed",
            "I32",
            &[2, 1],
            [0x87654321u32, 0xfedcba98]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        );
        add(
            "q.weight_scale",
            "F32",
            &[2, 1],
            [0.25f32, 0.5]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        );
        add(
            "q.weight_shape",
            "I64",
            &[2],
            [2i64, 8].into_iter().flat_map(i64::to_le_bytes).collect(),
        );
        add(
            "q.input_scale",
            "F32",
            &[8],
            [1f32; 8].into_iter().flat_map(f32::to_le_bytes).collect(),
        );
        add(
            "fp.weight",
            "F32",
            &[2, 8],
            [0.25f32; 16]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        );
        let mut header = serde_json::to_vec(&tensors)?;
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        std::fs::write(dir.path().join("model.safetensors"), file)?;
        std::fs::write(
            dir.path().join("bitwidth_manifest.json"),
            br#"{"format":"minicpmv47-c4-packed-v1","linear_bits":{"q":4}}"#,
        )?;
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"c4_quantization":{"format":"minicpmv47-c4-packed-v1"}}"#,
        )?;
        Ok(dir)
    }

    #[test]
    fn c4_metadata_uses_existing_packed_weight_math() -> Result<()> {
        let dir = c4_fixture()?;
        let c4 = QuantLoader::open(
            dir.path(),
            &serde_json::json!({"c4_quantization":{"format":"minicpmv47-c4-packed-v1"}}),
        )?;
        let old = QuantLoader::new(
            TensorStore::open(dir.path())?,
            &serde_json::json!({"quantization_config":{
            "quant_method":"compressed-tensors","quantization_status":"compressed","format":"pack-quantized",
            "config_groups":{"g":{"targets":["q"],
                "weights":{"num_bits":4,"type":"int","strategy":"channel","symmetric":true},
                "input_activations":{"num_bits":8,"type":"int","strategy":"token","symmetric":true,"dynamic":true}}}}}),
        )?;
        let actual = c4.load("q", 2, 8)?;
        assert_eq!(
            c4.source_stamps()?
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["bitwidth_manifest.json", "config.json", "model.safetensors"]
        );
        let baseline = old.load("q", 2, 8)?;
        assert_eq!(
            actual.codes,
            [-7, 0, -6, 1, -5, 2, -4, 3, -3, 4, -2, 5, -1, 6, 0, 7]
        );
        assert_eq!(actual.codes, baseline.codes);
        assert_eq!(actual.scales, [0.25, 0.5]);
        let row = c4.store.get("q.weight_packed")?.at(1)?;
        assert_eq!(row.shape, [1]);
        assert_eq!(row.data(), &0xfedcba98u32.to_le_bytes());
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive RK3588 NPU access"]
    fn c4_fused_projections_match_separate_and_preserve_float_fallback() -> Result<()> {
        use crate::model::{
            ModelOptions,
            cpu::Cpu,
            projection::{Linear, Projection},
        };

        let _lock = crate::validation::BoardLock::acquire()?;
        let dir = c4_fixture()?;
        let mut options = ModelOptions {
            path: dir.path().into(),
            context: 128,
            chunk: 128,
            threads: 4,
            // The fixture has only one padded N block.
            npu_mask: 1,
            cpu_mask: 0xf0,
            tune: false,
            tune_search: false,
            tune_cache: dir.path().into(),
            attention_tile: 512,
            vision_dtype: Default::default(),
            experts: Default::default(),
            speculation: Default::default(),
        };
        crate::npu::configure_cpu(&mut options)?;
        let backend = crate::npu::Backend::new(&options, 15)?;
        let cpu = Cpu::new(options.threads)?;
        let loader = Arc::new(QuantLoader::open(
            dir.path(),
            &serde_json::json!({"c4_quantization":{"format":"minicpmv47-c4-packed-v1"}}),
        )?);
        let separate = Linear::load(&backend, &loader, &cpu, "q", 2, 8)?;
        let fused = Projection::load(
            &backend,
            &loader,
            &cpu,
            vec![("q".into(), 2), ("q".into(), 2)],
            8,
        )?;
        // A fused projection needs no scatter scratch; separate modules do.
        assert_eq!(fused.scratch_width(), 0);
        let protected = Projection::load(
            &backend,
            &loader,
            &cpu,
            vec![("q".into(), 2), ("fp".into(), 2)],
            8,
        )?;
        assert_eq!(protected.scratch_width(), 2);
        let host = Linear::load(&backend, &loader, &cpu, "fp", 2, 8)?;
        for rows in [1, 5, 128] {
            backend.prepare(rows)?;
            let input: Vec<f32> = (0..rows * 8)
                .map(|i| ((i * 13 + i / 8 * 7) % 127) as f32 / 63. - 1.)
                .collect();
            let mut single = vec![0.; rows * 2];
            let mut floating = vec![0.; rows * 2];
            separate.run(&input, &mut single, rows)?;
            host.run(&input, &mut floating, rows)?;
            let mut actual = vec![0.; rows * 4];
            fused.run(&input, &mut actual, rows, &mut [])?;
            for r in 0..rows {
                assert_eq!(&actual[r * 4..r * 4 + 2], &single[r * 2..r * 2 + 2]);
                assert_eq!(&actual[r * 4 + 2..(r + 1) * 4], &single[r * 2..r * 2 + 2]);
            }
            protected.run(&input, &mut actual, rows, &mut vec![0.; rows * 2])?;
            for r in 0..rows {
                assert_eq!(&actual[r * 4..r * 4 + 2], &single[r * 2..r * 2 + 2]);
                assert_eq!(&actual[r * 4 + 2..(r + 1) * 4], &floating[r * 2..r * 2 + 2]);
            }
        }
        Ok(())
    }
}
