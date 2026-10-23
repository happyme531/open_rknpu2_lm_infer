//! Preserve XML function data before parsing; filter control tokens outside it.
use std::collections::HashSet;
use tokenizers::{
    Tokenizer, decoders::DecoderWrapper, models::ModelWrapper, normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper, processors::PostProcessorWrapper, tokenizer::DecodeStream,
};

pub(crate) struct OutputDecoder<'a> {
    inner: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    excluded: HashSet<u32>,
    framing: String,
    preserve_tool_markers: bool,
}
impl<'a> OutputDecoder<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, preserve_tool_markers: bool) -> Self {
        let excluded = if preserve_tool_markers {
            tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .filter(|(_, token)| {
                    token.special
                        && !matches!(
                            token.content.as_str(),
                            "<function" | "</function>" | "<param" | "</param>"
                        )
                })
                .map(|(id, _)| id)
                .collect()
        } else {
            HashSet::new()
        };
        Self {
            inner: tokenizer.decode_stream(!preserve_tool_markers),
            excluded,
            framing: String::new(),
            preserve_tool_markers,
        }
    }
    pub(crate) fn step(&mut self, token: u32) -> tokenizers::Result<Option<String>> {
        // Inside a function, even a token normally used as a control marker may
        // be literal CDATA/attribute data. EOS is handled by Engine before here.
        if self.excluded.contains(&token) && super::function_start(&self.framing) != Some(0) {
            return Ok(None);
        }
        let piece = self.inner.step(token)?;
        if self.preserve_tool_markers
            && let Some(piece) = &piece
        {
            self.framing.push_str(piece);
            loop {
                if let Some(start) = super::function_start(&self.framing) {
                    self.framing.drain(..start);
                    if let Some(end) = super::function_end(&self.framing) {
                        self.framing.drain(..end);
                        continue;
                    }
                } else {
                    let marker = "<function";
                    let hold = (1..=marker.len().min(self.framing.len()))
                        .rev()
                        .find(|n| {
                            self.framing.is_char_boundary(self.framing.len() - n)
                                && marker.starts_with(&self.framing[self.framing.len() - n..])
                        })
                        .unwrap_or(0);
                    self.framing.drain(..self.framing.len() - hold);
                }
                break;
            }
        }
        Ok(piece)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires MINICPM5_MODEL_PATH with real tokenizer.json"]
    fn contextual_filter_resets_after_calls_and_ignores_cdata_closings() {
        let path = std::path::PathBuf::from(std::env::var("MINICPM5_MODEL_PATH").unwrap());
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).unwrap();
        let calls = "<function name='f'><arguments><param name='s'><![CDATA[\"</function><tool_response>x</tool_response>\"]]></param></arguments></function>";
        let empty = "<function name='g'/>";
        let input = format!("<|im_start|>{calls}<tool_response>{empty}</tool_response><|im_end|>");
        let ids = tokenizer.encode(input, false).unwrap();
        let mut decoder = OutputDecoder::new(&tokenizer, true);
        let text: String = ids
            .get_ids()
            .iter()
            .filter_map(|&id| decoder.step(id).unwrap())
            .collect();
        assert_eq!(text, format!("{calls}{empty}"));
        // Default mode still delegates to the original skip-special decoder.
        let mut decoder = OutputDecoder::new(&tokenizer, false);
        let text: String = ids
            .get_ids()
            .iter()
            .filter_map(|&id| decoder.step(id).unwrap())
            .collect();
        assert_eq!(text, tokenizer.decode(ids.get_ids(), true).unwrap());
    }

    #[test]
    #[ignore = "requires MINICPM5_MODEL_PATH with real tokenizer.json"]
    fn real_minicpm5_tokenizer_retains_only_xml_markers() {
        let path = std::path::PathBuf::from(
            std::env::var("MINICPM5_MODEL_PATH").expect("set MINICPM5_MODEL_PATH"),
        );
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).unwrap();
        let text = "<function name=\"f\"><param name=\"城市\">北京</param></function>";
        let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let mut with_special = vec![tokenizer.token_to_id("<|im_start|>").unwrap()];
        with_special.extend(&ids);
        with_special.push(tokenizer.token_to_id("<|im_end|>").unwrap());
        for enabled in [false, true] {
            let mut decoder = OutputDecoder::new(&tokenizer, enabled);
            let result: String = with_special
                .iter()
                .filter_map(|&id| decoder.step(id).unwrap())
                .collect();
            if enabled {
                assert_eq!(result, text);
            } else {
                assert_eq!(result, tokenizer.decode(&with_special, true).unwrap());
            }
            assert!(!result.contains("<|im_"));
        }
    }
}
