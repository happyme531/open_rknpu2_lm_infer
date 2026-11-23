use crate::{
    sampling::Sampler,
    text::{OutputDecoder, StopFilter, TextCodec},
};
use anyhow::{Result, anyhow};

pub(super) struct GeneratedText<'a> {
    decoder: OutputDecoder<'a>,
    filter: StopFilter,
    limit: usize,
    generated: usize,
    stopped: bool,
    text: String,
}

impl<'a> GeneratedText<'a> {
    pub fn new(
        codec: &'a TextCodec,
        preserve_tool_markers: bool,
        limit: usize,
        stop: Vec<String>,
    ) -> Result<Self> {
        Ok(Self {
            decoder: OutputDecoder::new(&codec.tokenizer, preserve_tool_markers),
            filter: StopFilter::new(stop)?,
            limit,
            generated: 0,
            stopped: false,
            text: String::new(),
        })
    }

    pub fn generated(&self) -> usize {
        self.generated
    }

    pub fn reason(&self) -> Option<&'static str> {
        if self.stopped {
            Some("stop")
        } else {
            (self.generated == self.limit).then_some("length")
        }
    }

    pub fn push(
        &mut self,
        token: u32,
        eos: &[u32],
        sampler: &mut Sampler,
    ) -> Result<Option<String>> {
        debug_assert!(self.reason().is_none());
        self.generated += 1;
        if eos.contains(&token) {
            self.stopped = true;
            return Ok(None);
        }
        sampler.observe(token);
        let Some(piece) = self
            .decoder
            .step(token)
            .map_err(|e| anyhow!("stream decode: {e}"))?
        else {
            return Ok(None);
        };
        let (text, stopped) = self.filter.push(&piece);
        self.stopped = stopped;
        self.text.push_str(&text);
        Ok((!text.is_empty()).then_some(text))
    }

    pub fn flush(&mut self) -> String {
        let tail = self.filter.finish();
        self.text.push_str(&tail);
        tail
    }

    pub fn into_text(self) -> String {
        self.text
    }
}
