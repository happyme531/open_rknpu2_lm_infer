use super::{GenerationProgress, Phase, take_notice};
use crossterm::{
    cursor, execute,
    terminal::{self, Clear, ClearType},
};
use ratatui::text::Span;
use std::{
    io::{self, IsTerminal, Write},
    time::{Duration, Instant},
};

#[derive(Default)]
struct RenderedStatus {
    text: String,
    rows: u16,
}
impl RenderedStatus {
    fn new(state: &GenerationProgress, notice: Option<&str>, width: usize) -> Self {
        let mut lines = state.lines();
        if let Some(notice) = notice {
            lines.insert(0, format!("Warning: {notice}"));
        }
        let lines = wrap_status(&lines, width.saturating_sub(1).max(1));
        Self {
            text: lines.join("\r\n"),
            rows: lines.len() as u16,
        }
    }
    fn write(&self, output: &mut impl Write, permanent: bool) -> io::Result<()> {
        output.write_all(self.text.as_bytes())?;
        if permanent {
            output.write_all(b"\r\n")?;
        }
        output.flush()
    }
}

/// The live rows sit below streamed stdout and are removed before appending
/// text. Keeping the cursor in the status area also works at the screen bottom.
pub(crate) struct TerminalProgress {
    terminal: bool,
    interactive: bool,
    inline_text: bool,
    rows: u16,
    column: usize,
    width: usize,
    has_text: bool,
    last: Instant,
    phase: Option<Phase>,
    snapshot: GenerationProgress,
    notice: Option<String>,
    rendered: RenderedStatus,
}
impl TerminalProgress {
    pub fn new(inline_text: bool) -> Self {
        let interactive = io::stderr().is_terminal();
        // Server logs use complete lines, so tracing and startup messages can
        // share stderr without invalidating cursor positions.
        let terminal = interactive && inline_text && !tracing::enabled!(tracing::Level::INFO);
        Self {
            terminal,
            interactive,
            inline_text: inline_text && terminal && io::stdout().is_terminal(),
            rows: 0,
            column: 0,
            width: terminal::size().map_or(80, |s| s.0.max(2) as usize),
            has_text: false,
            last: Instant::now(),
            phase: None,
            snapshot: GenerationProgress::default(),
            notice: None,
            rendered: RenderedStatus::default(),
        }
    }
    fn clear(&mut self) -> io::Result<()> {
        if self.rows > 0 {
            let mut err = io::stderr().lock();
            execute!(err, cursor::MoveToColumn(0))?;
            if self.rows > 1 {
                execute!(err, cursor::MoveUp(self.rows - 1))?;
            }
            execute!(err, Clear(ClearType::FromCursorDown))?;
            self.rows = 0;
        }
        Ok(())
    }
    fn draw(&mut self, permanent: bool) -> io::Result<()> {
        let mut err = io::stderr().lock();
        self.rendered.write(&mut err, permanent)?;
        self.rows = if permanent { 0 } else { self.rendered.rows };
        Ok(())
    }
    fn render(&mut self) {
        self.rendered = RenderedStatus::new(&self.snapshot, self.notice.as_deref(), self.width);
    }
    pub fn update(&mut self, state: &GenerationProgress, force: bool) -> io::Result<()> {
        let mut notice_changed = false;
        if let Some(message) = take_notice() {
            notice_changed = true;
            // Full warnings remain readable even on narrow terminals.
            if !self.has_text || !self.terminal {
                self.clear()?;
                eprintln!("Warning: {message}");
            } else {
                self.notice = Some(message);
            }
        }
        let finished = matches!(
            state.phase,
            Phase::Finished | Phase::Cancelled | Phase::Error
        );
        let changed = self.phase != Some(state.phase) || self.snapshot.request != state.request;
        let interval = if self.terminal {
            Duration::from_millis(250)
        } else if self.interactive {
            Duration::from_secs(1)
        } else {
            Duration::from_secs(5)
        };
        self.snapshot = state.clone();
        if !force && !changed && !notice_changed && self.last.elapsed() < interval {
            return Ok(());
        }
        self.phase = Some(state.phase);
        self.last = Instant::now();
        if self.terminal {
            self.clear()?;
            self.render();
            self.draw(finished)?;
        } else {
            eprintln!("[request {}] {}", state.request, state.lines().join(" · "));
        }
        Ok(())
    }
    pub fn token(&mut self, text: &str) -> io::Result<()> {
        if !self.inline_text {
            let mut out = io::stdout().lock();
            write!(out, "{text}")?;
            return out.flush();
        }
        self.clear()?;
        let width = terminal::size().map_or(self.width, |s| s.0.max(2) as usize);
        if self.has_text && self.width == width && self.column < width {
            execute!(
                io::stderr(),
                cursor::MoveUp(1),
                cursor::MoveToColumn(self.column as u16)
            )?;
        } else {
            self.column = 0;
        }
        if self.width != width {
            self.width = width;
            self.render();
        }
        {
            let mut out = io::stdout().lock();
            write!(out, "{text}")?;
            out.flush()?;
        }
        for ch in text.chars() {
            match ch {
                '\n' | '\r' => self.column = 0,
                '\t' => self.column = ((self.column / 8 + 1) * 8).min(width - 1),
                '\u{8}' => self.column = self.column.saturating_sub(1),
                _ => {
                    let size = Span::raw(ch.to_string()).width();
                    self.column = if self.column + size > width {
                        size
                    } else {
                        self.column + size
                    };
                }
            }
        }
        write!(io::stderr(), "\r\n")?;
        self.has_text = true;
        self.draw(false)
    }
    pub fn finish(&mut self, state: &GenerationProgress) -> io::Result<()> {
        self.update(state, true)?;
        self.has_text = false;
        Ok(())
    }
}

fn wrap_status(lines: &[String], width: usize) -> Vec<String> {
    let mut result = Vec::new();
    for source in lines {
        let mut line = String::new();
        let mut used = 0;
        for ch in source.chars() {
            let size = Span::raw(ch.to_string()).width();
            if used + size > width && !line.is_empty() {
                result.push(std::mem::take(&mut line));
                used = 0;
            }
            line.push(ch);
            used += size;
        }
        result.push(line);
    }
    result
}
impl Drop for TerminalProgress {
    fn drop(&mut self) {
        let _ = self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Output {
        writes: Vec<Vec<u8>>,
        flushes: usize,
    }
    impl Write for Output {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes.push(bytes.to_vec());
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn cached_status_redraws_use_one_payload_write_and_keep_the_rendered_snapshot() {
        let mut state = GenerationProgress {
            phase: Phase::Decode,
            ..Default::default()
        };
        state.decode.generated = 4;
        let rendered = RenderedStatus::new(&state, None, 80);
        state.decode.generated = 5;
        let mut output = Output::default();
        rendered.write(&mut output, false).unwrap();
        rendered.write(&mut output, false).unwrap();
        assert_eq!(output.writes.len(), 2);
        assert_eq!(output.flushes, 2);
        assert_eq!(output.writes[0], output.writes[1]);
        assert!(rendered.text.contains("4 output"));
        assert!(
            RenderedStatus::new(&state, None, 80)
                .text
                .contains("5 output")
        );
    }

    #[test]
    fn status_cache_rebuilds_wrapping_for_a_new_width_or_notice() {
        let state = GenerationProgress {
            phase: Phase::Decode,
            speculation: "draft-mtp",
            ..Default::default()
        };
        let wide = RenderedStatus::new(&state, None, 100);
        let narrow = RenderedStatus::new(&state, Some("未调优的 MTP projection"), 48);
        assert!(narrow.rows > wide.rows);
        assert_eq!(narrow.text.split("\r\n").count(), narrow.rows as usize);
        assert!(narrow.text.contains("未调优"));
        for line in narrow.text.split("\r\n") {
            assert!(Span::raw(line).width() <= 47);
        }
        let mut output = Output::default();
        narrow.write(&mut output, true).unwrap();
        assert_eq!(output.writes.len(), 2);
        assert_eq!(output.writes[1], b"\r\n");
    }

    #[test]
    #[ignore = "requires a TTY"]
    fn tty_stream_uses_cached_status() -> io::Result<()> {
        let mut display = TerminalProgress::new(true);
        assert!(
            display.inline_text,
            "run this test under a PTY with --nocapture"
        );
        println!();
        let mut state = GenerationProgress {
            phase: Phase::Decode,
            speculation: "draft-mtp",
            ..Default::default()
        };
        display.update(&state, true)?;
        let rendered = display.rendered.text.clone();
        let body = include_str!("../../tests/fixtures/progress_stream.txt");
        for (index, chunk) in body.as_bytes().chunks(11).enumerate() {
            state.decode.generated = index + 1;
            // Keep this update inside the refresh interval regardless of PTY latency.
            display.last = Instant::now();
            display.update(&state, false)?;
            display.token(std::str::from_utf8(chunk).unwrap())?;
            assert_eq!(display.rendered.text, rendered);
        }
        state.phase = Phase::Finished;
        display.finish(&state)
    }
}
