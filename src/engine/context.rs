//! Prefix identity includes image content, patch geometry and all RoPE axes.
use crate::{
    input::{PreparedInput, RowIdentity},
    model::ModelInfo,
};
use anyhow::{Result, ensure};

#[derive(Default)]
pub(super) struct PrefixCache {
    rows: Vec<RowIdentity>,
    next_position: usize,
    pub logits: Vec<f32>,
}
impl PrefixCache {
    #[cfg(test)]
    pub(super) fn rows_len(&self) -> usize {
        self.rows.len()
    }
    pub fn clear(&mut self) {
        self.rows.clear();
        self.logits.clear();
        self.next_position = 0;
    }
    pub fn truncate(&mut self, length: usize) {
        self.rows.truncate(length);
    }
    pub fn reusable(&self, input: &PreparedInput, position: usize) -> usize {
        if self.rows.len() != position || input.tokens.is_empty() {
            return 0;
        }
        self.rows
            .iter()
            .enumerate()
            .take(input.tokens.len())
            .take_while(|(i, row)| **row == input.row(*i))
            .count()
    }
    pub fn extend_input(&mut self, input: &PreparedInput, start: usize) {
        self.rows
            .extend((start..input.tokens.len()).map(|i| input.row(i)));
        self.next_position = input.next_position;
    }
    pub fn append_token(&mut self, token: u32) {
        self.rows.push(RowIdentity {
            token,
            position: [self.next_position; 3],
            image: None,
        });
        self.next_position += 1;
    }
}
pub(super) fn validate_request_tokens(
    tokens: &[u32],
    max_tokens: usize,
    info: &ModelInfo,
) -> Result<()> {
    ensure!(!tokens.is_empty(), "prompt must contain at least one token");
    ensure!(
        tokens.iter().all(|&t| (t as usize) < info.vocab_size),
        "prompt contains a token outside the vocabulary"
    );
    ensure!(
        tokens.len() <= info.context && max_tokens <= info.context - tokens.len(),
        "requested {} prompt + {} completion tokens exceed context {}",
        tokens.len(),
        max_tokens,
        info.context
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_positions_and_image_identity_are_required() {
        let mut cache = PrefixCache::default();
        cache.extend_input(&PreparedInput::text(vec![1, 2, 3]), 0);
        assert_eq!(cache.reusable(&PreparedInput::text(vec![1, 2, 3, 4]), 3), 3);
        assert_eq!(cache.reusable(&PreparedInput::text(vec![1, 2, 4]), 3), 2);
        assert_eq!(cache.reusable(&PreparedInput::text(vec![1, 2]), 3), 2);
        assert_eq!(cache.reusable(&PreparedInput::text(vec![1, 2, 3]), 2), 0);
        let mut input = PreparedInput::text(vec![1, 2, 3, 4]);
        input.positions[1] = [9; 3];
        assert_eq!(cache.reusable(&input, 3), 1);
        input.positions[1] = [1; 3];
        input.images.push(crate::input::ImageSpan {
            start: 1,
            image: std::sync::Arc::new(crate::vision::PreparedImage {
                identity: [1; 32],
                grid: [1, 2, 2],
                patches: vec![0.0; 4 * 1536].into(),
            }),
        });
        cache.clear();
        cache.extend_input(&input, 0);
        let mut other = input.clone();
        other.images[0].image = std::sync::Arc::new(crate::vision::PreparedImage {
            identity: [2; 32],
            ..(*input.images[0].image).clone()
        });
        assert_eq!(cache.reusable(&other, 4), 1);
    }
    #[test]
    fn context_reserves_completion_before_queueing() {
        let info = ModelInfo {
            id: "test".into(),
            architecture: "test".into(),
            context: 4,
            vocab_size: 10,
            eos_tokens: vec![9],
        };
        assert!(validate_request_tokens(&[1, 2], 2, &info).is_ok());
        assert!(validate_request_tokens(&[1, 2], 3, &info).is_err());
        assert!(validate_request_tokens(&[10], 0, &info).is_err());
    }
}
