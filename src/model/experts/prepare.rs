//! Prepare native records one layer at a time, without a resident model.
use super::store::{self, Geometry, Store, Writer};
use crate::{
    model::{ModelOptions, checkpoint::SourceStamp},
    npu::{Backend, QuantMatrix},
};
use anyhow::{Result, ensure};
use std::{path::Path, sync::Arc};

pub fn open(
    directory: &Path,
    geometry: Geometry,
    options: &ModelOptions,
    sources: impl Fn() -> Result<Vec<SourceStamp>>,
    mut load: impl FnMut(usize, usize) -> Result<(QuantMatrix, QuantMatrix)>,
) -> Result<Arc<Store>> {
    let stamps = sources()?;
    if let Ok(store) = Store::open(directory, &geometry, &stamps, false) {
        return Ok(Arc::new(store.with_direct(options.experts.direct)?));
    }
    let _lock = store::lock(directory)?;
    if let Ok(store) = Store::open(directory, &geometry, &stamps, false) {
        return Ok(Arc::new(store.with_direct(options.experts.direct)?));
    }
    let mut writer = None;
    for layer in 0..geometry.layers {
        let backend = Backend::new(options, 1)?;
        let mut gate = Vec::with_capacity(geometry.experts);
        let mut down = Vec::with_capacity(geometry.experts);
        for expert in 0..geometry.experts {
            let (g, d) = load(layer, expert)?;
            gate.push(g);
            down.push(d);
        }
        let bank = backend.load_moe(&gate, &down, 1)?;
        drop((gate, down));
        if writer.is_none() {
            writer = Some(Writer::create(
                directory,
                geometry.clone(),
                stamps.clone(),
                bank.blob_bytes(),
            )?);
        }
        for expert in 0..geometry.experts {
            writer
                .as_mut()
                .unwrap()
                .write(layer, expert, &bank.export(expert)?)?;
        }
        eprintln!("prepared expert records {}/{}", layer + 1, geometry.layers);
    }
    ensure!(
        sources()? == stamps,
        "checkpoint changed during expert preparation"
    );
    writer.expect("nonempty expert geometry").finish()?;
    Ok(Arc::new(Store::open(
        directory,
        &geometry,
        &stamps,
        options.experts.direct,
    )?))
}
