use crate::{
    engine::{Engine, GenerationEvent, GenerationHandle, GenerationRequest, Prompt},
    sampling::SamplingParams,
};
use anyhow::Result;
use crossterm::{
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind,
        KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
};
use serde_json::{Value, json};
use std::{io, sync::atomic::Ordering, time::Duration};

struct TerminalGuard;
impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let guard = Self;
        execute!(io::stdout(), EnterAlternateScreen, EnableBracketedPaste)?;
        Ok(guard)
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), DisableBracketedPaste, LeaveAlternateScreen);
    }
}
struct Conversation {
    messages: Vec<Value>,
    transcript: Vec<(String, String)>,
    input: String,
    pending_images: Vec<String>,
    active: Option<GenerationHandle>,
    answer: String,
    status: String,
    scroll: u16,
    follow: bool,
}
fn wrap_lines(text: &str, width: usize) -> Vec<Line<'static>> {
    let mut lines = vec![];
    for source in text.split('\n') {
        let mut current = String::new();
        let mut used = 0;
        for character in source.chars() {
            let character_width = Span::raw(character.to_string()).width();
            if used + character_width > width && !current.is_empty() {
                lines.push(Line::from(std::mem::take(&mut current)));
                used = 0;
            }
            current.push(character);
            used += character_width;
        }
        lines.push(Line::from(current));
    }
    lines
}
impl Conversation {
    fn reset(&mut self, system: &Option<String>) {
        self.active.take();
        self.messages.clear();
        self.pending_images.clear();
        self.transcript.clear();
        self.answer.clear();
        self.scroll = 0;
        self.follow = true;
        if let Some(system) = system {
            self.messages
                .push(json!({"role":"system","content":system}));
        }
        self.status = "Ready".into();
    }
    fn finish(&mut self) {
        if !self.answer.is_empty() {
            self.messages
                .push(json!({"role":"assistant","content":self.answer}));
        }
        self.answer.clear();
        self.active.take();
    }
    fn send(
        &mut self,
        engine: &Engine,
        thinking: bool,
        sampling: &SamplingParams,
        max_tokens: usize,
        stop: &[String],
    ) {
        let input = self.input.trim().to_owned();
        if input.is_empty() || self.active.is_some() {
            return;
        }
        self.input.clear();
        if let Some(path) = input.strip_prefix("/image ") {
            if self.pending_images.len() >= crate::vision::MAX_IMAGES {
                self.status = "At most 8 images per message".into();
                return;
            }
            match crate::vision::data_url(std::path::Path::new(path.trim())) {
                Ok(url) => {
                    self.pending_images.push(url);
                    self.status = format!(
                        "{} image(s) attached to next message",
                        self.pending_images.len()
                    );
                }
                Err(e) => self.status = format!("Image error: {e}"),
            }
            return;
        }
        let content = if self.pending_images.is_empty() {
            json!(input)
        } else {
            let mut parts: Vec<_> = self
                .pending_images
                .iter()
                .map(|url| json!({"type":"image_url","image_url":{"url":url}}))
                .collect();
            parts.push(json!({"type":"text","text":input}));
            json!(parts)
        };
        self.messages.push(json!({"role":"user","content":content}));
        self.transcript.push((
            "You".into(),
            format!(
                "{}{}",
                if self.pending_images.is_empty() {
                    String::new()
                } else {
                    format!("[{} image(s)] ", self.pending_images.len())
                },
                input
            ),
        ));
        match engine.generate(GenerationRequest {
            prompt: Prompt::Chat {
                messages: self.messages.clone(),
                tools: None,
                enable_thinking: thinking,
            },
            sampling: sampling.clone(),
            max_tokens,
            stop: stop.to_vec(),
        }) {
            Ok(handle) => {
                self.pending_images.clear();
                self.active = Some(handle);
                self.answer.clear();
                self.transcript.push(("Assistant".into(), String::new()));
                self.status = "Generating · Esc to cancel".into();
                self.follow = true;
            }
            Err(error) => {
                self.messages.pop();
                self.status = format!("Error: {error}");
            }
        }
    }
    fn receive(&mut self) {
        loop {
            let event = match self.active.as_mut() {
                Some(handle) => handle.events.try_recv(),
                None => return,
            };
            match event {
                Ok(GenerationEvent::Token(text)) => {
                    self.answer.push_str(&text);
                    if let Some((_, answer)) = self.transcript.last_mut() {
                        answer.push_str(&text);
                    }
                }
                Ok(GenerationEvent::Finished(result)) => {
                    self.status = format!(
                        "{} prompt · {} cached · {} output · {}",
                        result.prompt_tokens,
                        result.cached_tokens,
                        result.completion_tokens,
                        result.finish_reason
                    );
                    self.finish();
                    return;
                }
                Ok(GenerationEvent::Error(message)) => {
                    self.status = format!("Error: {message}");
                    self.finish();
                    return;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    self.status = "Generation stopped".into();
                    self.finish();
                    return;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => return,
            }
        }
    }
}
pub async fn chat(
    engine: Engine,
    system: Option<String>,
    thinking: bool,
    sampling: SamplingParams,
    max_tokens: usize,
    stop: Vec<String>,
) -> Result<()> {
    let _guard = TerminalGuard::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;
    let mut app = Conversation {
        messages: vec![],
        transcript: vec![],
        input: String::new(),
        pending_images: Vec::new(),
        active: None,
        answer: String::new(),
        status: String::new(),
        scroll: 0,
        follow: true,
    };
    app.reset(&system);
    let mut tick = tokio::time::interval(Duration::from_millis(40));
    loop {
        app.receive();
        terminal.draw(|frame| {
            let area = frame.area();
            let input_width = area.width.saturating_sub(2).max(1) as usize;
            let mut input_lines = wrap_lines(&app.input, input_width);
            if input_lines
                .last()
                .is_some_and(|line| line.width() >= input_width)
            {
                input_lines.push(Line::from(""));
            }
            let input_height = (input_lines.len().min(u16::MAX as usize) as u16)
                .saturating_add(2)
                .clamp(3, (area.height / 3).max(3));
            let panes = Layout::vertical([
                Constraint::Length(2),
                Constraint::Min(1),
                Constraint::Length(input_height),
                Constraint::Length(1),
            ])
            .split(area);
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        format!(" ◆ {} ", crate::PROJECT_NAME),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(engine.info().id.clone(), Style::default().fg(Color::White)),
                    Span::styled(
                        if thinking { " · thinking" } else { "" },
                        Style::default().fg(Color::DarkGray),
                    ),
                ])),
                panes[0],
            );
            let mut lines = vec![];
            let width = panes[1].width.saturating_sub(2).max(1) as usize;
            if app.transcript.is_empty() {
                lines.push(Line::styled(
                    " Send a message to begin.",
                    Style::default().fg(Color::DarkGray),
                ));
            }
            for (role, text) in &app.transcript {
                lines.push(Line::styled(
                    role.to_string(),
                    Style::default()
                        .fg(if role == "You" {
                            Color::Green
                        } else {
                            Color::Cyan
                        })
                        .add_modifier(Modifier::BOLD),
                ));
                lines.extend(wrap_lines(text, width));
                lines.push(Line::from(""));
            }
            let text = Text::from(lines);
            let rows = text.lines.len();
            let visible = panes[1].height.saturating_sub(2) as usize;
            let maximum_scroll = rows.saturating_sub(visible).min(u16::MAX as usize) as u16;
            if app.follow {
                app.scroll = maximum_scroll;
            } else {
                app.scroll = app.scroll.min(maximum_scroll);
            }
            frame.render_widget(
                Paragraph::new(text)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::DarkGray)),
                    )
                    .scroll((app.scroll, 0)),
                panes[1],
            );
            let column = input_lines.last().map(Line::width).unwrap_or(0);
            let row = input_lines.len().saturating_sub(1);
            let input_scroll = input_lines
                .len()
                .saturating_sub(panes[2].height.saturating_sub(2) as usize);
            frame.render_widget(
                Paragraph::new(Text::from(input_lines))
                    .block(
                        Block::default()
                            .title(" Message · Enter send · Shift+Enter newline ")
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(Color::Cyan)),
                    )
                    .scroll((input_scroll.min(u16::MAX as usize) as u16, 0)),
                panes[2],
            );
            let footer = format!(
                "{} · /image PATH · /reset · PgUp/PgDn · Ctrl+C exit",
                app.status
            );
            frame.render_widget(
                Paragraph::new(footer).style(Style::default().fg(Color::DarkGray)),
                panes[3],
            );
            if panes[2].width > 2 && panes[2].height > 2 {
                frame.set_cursor_position((
                    panes[2].x + 1 + (column % panes[2].width.saturating_sub(2) as usize) as u16,
                    panes[2].y
                        + 1
                        + (row
                            .saturating_sub(input_scroll)
                            .min((panes[2].height - 3) as usize) as u16),
                ));
            }
        })?;
        while event::poll(Duration::ZERO)? {
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    KeyCode::Esc => {
                        if let Some(handle) = app.active.as_ref() {
                            handle.cancel.store(true, Ordering::Relaxed);
                        }
                        app.status = "Cancelled".into();
                        app.finish();
                    }
                    KeyCode::Enter
                        if key.modifiers.contains(KeyModifiers::SHIFT)
                            || key.modifiers.contains(KeyModifiers::ALT) =>
                    {
                        app.input.push('\n')
                    }
                    KeyCode::Enter => {
                        if app.input.trim() == "/reset" {
                            app.input.clear();
                            app.reset(&system);
                        } else {
                            app.send(&engine, thinking, &sampling, max_tokens, &stop);
                        }
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.input.clear()
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.input.push(c)
                    }
                    KeyCode::PageUp => {
                        app.follow = false;
                        app.scroll = app.scroll.saturating_sub(10);
                    }
                    KeyCode::PageDown => {
                        app.scroll = app.scroll.saturating_add(10);
                        app.follow = false;
                    }
                    KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.follow = true
                    }
                    _ => (),
                },
                Event::Paste(text) => app.input.push_str(&text),
                _ => (),
            }
        }
        tick.tick().await;
    }
}
