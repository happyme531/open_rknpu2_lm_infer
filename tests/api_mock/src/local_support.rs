mod vision {
    pub struct ImageProcessor;
    impl ImageProcessor {
        pub fn load(_: &std::path::Path) -> anyhow::Result<Option<Self>> {
            Ok(None)
        }
    }
}
mod input {
    pub struct PreparedInput {
        pub tokens: Vec<u32>,
        pub user_starts: Vec<usize>,
    }
    impl PreparedInput {
        pub fn text(tokens: Vec<u32>) -> Self {
            Self {
                tokens,
                user_starts: Vec::new(),
            }
        }
        pub fn set_user_starts(&mut self, positions: &[usize]) {
            self.user_starts = positions.to_vec();
        }
        pub fn expand(
            _: Vec<u32>,
            _: Vec<()>,
            _: &crate::vision::ImageProcessor,
        ) -> anyhow::Result<Self> {
            anyhow::bail!("images unavailable in API mock")
        }
    }
    pub fn message_images(
        _: &[serde_json::Value],
        _: Option<&crate::vision::ImageProcessor>,
    ) -> anyhow::Result<Vec<()>> {
        Ok(vec![])
    }
}
