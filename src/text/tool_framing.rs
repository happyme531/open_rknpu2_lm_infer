//! Shared XML framing for protocol parsing and context-aware token decoding.
pub(crate) fn function_start(text: &str) -> Option<usize> {
    text.match_indices("<function").find_map(|(i, _)| {
        text.as_bytes()
            .get(i + 9)
            .filter(|c| c.is_ascii_whitespace() || **c == b'>')
            .map(|_| i)
    })
}

// Find a complete outer function, ignoring closing-tag spellings inside CDATA,
// comments and quoted attributes. Nested functions remain one invalid block.
pub(crate) fn function_end(text: &str) -> Option<usize> {
    let mut pos = 0;
    let mut depth = 0usize;
    while let Some(offset) = text[pos..].find('<') {
        let start = pos + offset;
        let rest = &text[start..];
        if let Some(body) = rest.strip_prefix("<![CDATA[") {
            pos = start + 9 + body.find("]]>")? + 3;
            continue;
        }
        if let Some(body) = rest.strip_prefix("<!--") {
            pos = start + 4 + body.find("-->")? + 3;
            continue;
        }
        let mut quote = None;
        let end = rest.bytes().enumerate().find_map(|(i, c)| {
            if let Some(q) = quote {
                if c == q {
                    quote = None;
                }
            } else if c == b'\'' || c == b'"' {
                quote = Some(c);
            } else if c == b'>' {
                return Some(start + i + 1);
            }
            None
        })?;
        let tag = &text[start..end];
        if tag.starts_with("</function") && tag[10..tag.len() - 1].trim().is_empty() {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(end);
            }
        } else if function_start(tag) == Some(0) {
            depth += 1;
            if tag.ends_with("/>") {
                depth -= 1;
                if depth == 0 {
                    return Some(end);
                }
            }
        }
        pos = end;
    }
    None
}
