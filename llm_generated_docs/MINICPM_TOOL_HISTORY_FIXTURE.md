# MiniCPM-V 4.7 tool-call history fixture

`tests/fixtures/minicpmv47_tool_history/chat_template.jinja` is the template of
`happyme531/MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT` at revision
`21fe85b0556e0235c1efca61060cecac3a41ed05` (Apache-2.0), SHA-256
`c4531cacbf2a0ab65170433d7a4f371d11b7599cf96c04b192e6372432636c70`.

`case.json` and `expected.txt` were rendered by transformers
5.3.0 (`render_jinja_template`, thinking off). That is the reference the
`minicpmv47_*` tests in `src/api.rs` compare against: tool-call argument
keys keep their order, and `tojson` is `json.dumps`.
