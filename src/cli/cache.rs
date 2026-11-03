use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

pub(super) fn default(model: &Path) -> PathBuf {
    resolve(
        model,
        std::env::var_os("CACHE_DIRECTORY").as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

fn resolve(
    model: &Path,
    service: Option<&OsStr>,
    xdg: Option<&OsStr>,
    home: Option<&OsStr>,
) -> PathBuf {
    if let Some(path) = service
        .and_then(|value| std::env::split_paths(value).find(|path| !path.as_os_str().is_empty()))
    {
        return path;
    }
    xdg.filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            home.filter(|value| !value.is_empty())
                .map(|home| PathBuf::from(home).join(".cache"))
        })
        .unwrap_or_else(|| model.join(".cache"))
        .join(crate::PROJECT_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_cache_uses_the_configured_directory() {
        assert_eq!(
            resolve(
                Path::new("model"),
                Some(OsStr::new("/var/cache/infer:/var/cache/other")),
                Some(OsStr::new("/xdg")),
                Some(OsStr::new("/home/user")),
            ),
            Path::new("/var/cache/infer")
        );
    }

    #[test]
    fn interactive_cache_keeps_xdg_then_home_precedence() {
        let model = Path::new("model");
        let home = Some(OsStr::new("/home/user"));
        assert_eq!(
            resolve(model, None, Some(OsStr::new("/xdg")), home),
            Path::new("/xdg").join(crate::PROJECT_NAME)
        );
        assert_eq!(
            resolve(model, None, None, home),
            Path::new("/home/user/.cache").join(crate::PROJECT_NAME)
        );
    }

    #[test]
    fn missing_or_empty_environment_uses_a_persistent_model_cache() {
        let model = Path::new("/models/qwen");
        let expected = model.join(".cache").join(crate::PROJECT_NAME);
        assert_eq!(resolve(model, None, None, None), expected);
        assert_eq!(
            resolve(
                model,
                Some(OsStr::new("")),
                Some(OsStr::new("")),
                Some(OsStr::new("")),
            ),
            expected
        );
    }
}
