mod vision {
    pub struct ImageProcessor;
    impl ImageProcessor {
        pub fn load(_: &std::path::Path) -> anyhow::Result<Option<Self>> {
            Ok(None)
        }
    }
}
mod input {
    pub struct PreparedInput;
    impl PreparedInput {
        pub fn text(_: Vec<u32>) -> Self {
            Self
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
