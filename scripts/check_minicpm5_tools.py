#!/usr/bin/env python3
"""Prompt-guided MiniCPM5 tool checks against the RK3588 server.

No tools are executed. API shape and expected model values are scored separately;
a successful API response alone cannot establish that the parser was lossless.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from xml.sax.saxutils import escape


def same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def reject_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


def loads(value):
    return json.loads(value, parse_constant=reject_constant)


def xml_call(name, arguments):
    # Quote JSON strings too, so "123" and "null" stay strings under literal decoding.
    params = ''.join(f'<param name="{key}">{escape(dumps(value))}</param>'
                     for key, value in arguments.items())
    return f'<function name="{name}">{params}</function>'


def cases():
    return [
        ("scalar_types", [("capture", {"text": "北京", "integer": 123,
            "fraction": 1.25, "flag": True, "empty": None})]),
        ("string_literals", [("capture", {"number_text": "123", "null_text": "null",
            "bool_text": "true", "blank": "", "spaced": "  keep\n "})]),
        ("nested", [("capture", {"items": [1, "x", False, None],
            "config": {"count": 2, "enabled": True}, "literal": "<param>&value"})]),
        ("consecutive_calls", [("capture", {"step": 1}), ("record", {"step": 2})]),
        ("tool_feedback", [("capture", {"ticket": "board-check"})]),
    ]


def score(response, expected):
    try:
        choice = response["choices"][0]
        message = choice["message"]
        calls = message["tool_calls"]
        if not isinstance(calls, list) or not calls or choice["finish_reason"] != "tool_calls":
            raise ValueError("missing tool_calls finish or calls")
        decoded = []
        ids = set()
        for call in calls:
            if call["type"] != "function" or not isinstance(call["id"], str) or not call["id"] or call["id"] in ids:
                raise ValueError("invalid call type or ID")
            ids.add(call["id"])
            function = call["function"]
            if not isinstance(function["name"], str) or not function["name"] or not isinstance(function["arguments"], str):
                raise ValueError("invalid function fields")
            args = loads(function["arguments"])
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            decoded.append((function["name"], args))
        matches = len(decoded) == len(expected) and all(
            name == want_name and same(args, want_args)
            for (name, args), (want_name, want_args) in zip(decoded, expected))
        return {"api_shape_ok": True, "expected_values_ok": matches}
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {"api_shape_ok": False, "expected_values_ok": False, "error": str(error)}


def assemble_sse(raw):
    """Reassemble OpenAI deltas for scoring; preserve raw SSE separately as evidence."""
    message = {'role': 'assistant', 'content': ''}
    calls = {}
    finish = None
    usage = None
    done = False
    for event in raw.replace('\r\n', '\n').split('\n\n'):
        data = '\n'.join(line[5:].lstrip(' ') for line in event.splitlines() if line.startswith('data:'))
        if not data:
            continue
        if done:
            raise ValueError('SSE event after DONE')
        if data == '[DONE]':
            done = True
            continue
        chunk = loads(data)
        if 'error' in chunk:
            raise ValueError('SSE server error: ' + dumps(chunk['error']))
        if chunk.get('usage') is not None:
            usage = chunk['usage']
        for choice in chunk['choices']:
            if choice['index'] != 0:
                raise ValueError('unexpected SSE choice index')
            delta = choice['delta']
            for key in ['content', 'reasoning_content']:
                if delta.get(key) is not None:
                    message[key] = message.get(key, '') + delta[key]
            for call in delta.get('tool_calls', []):
                index = call['index']
                if type(index) is not int or index < 0:
                    raise ValueError('invalid SSE tool index')
                target = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                for key in ['id', 'type']:
                    if key in call:
                        if key == 'id':
                            target[key] += call[key]
                        else:
                            target[key] = call[key]
                for key in ['name', 'arguments']:
                    target['function'][key] += call.get('function', {}).get(key, '')
            if choice.get('finish_reason') is not None:
                if finish is not None:
                    raise ValueError('duplicate SSE finish')
                finish = choice['finish_reason']
    if not done or finish is None:
        raise ValueError('incomplete SSE response')
    if calls:
        if sorted(calls) != list(range(len(calls))):
            raise ValueError('non-contiguous SSE tool indices')
        message['tool_calls'] = [calls[i] for i in sorted(calls)]
    return {'choices': [{'message': message, 'finish_reason': finish}], 'usage': usage}


def http(base, key, endpoint, timeout, payload=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(base.rstrip('/') + endpoint,
        data=dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            if payload and payload.get('stream'):
                try:
                    parsed = assemble_sse(raw)
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    parsed = {'error': str(error)}
                return response.status, raw, parsed
            return response.status, raw, loads(raw)
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        return error.code, raw, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://192.168.10.119:8080/v1')
    parser.add_argument('--api-key', default=os.environ.get('OPEN_RKNPU_API_KEY', ''))
    parser.add_argument('--model')
    parser.add_argument('--model-revision', default='unrecorded')
    parser.add_argument('--template-path', type=Path, help='Exact template used by server, for SHA-256 provenance')
    parser.add_argument('--server-revision', default='unrecorded', help='Commit plus dirty-diff identity of tested server')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--stream', action='store_true', help='Request SSE and score reconstructed tool calls; raw SSE is saved')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    report = {
        'evidence': 'RK3588 prompt-guided real inference; no tool execution',
        'target_host': '192.168.10.119', 'base_url': args.base_url,
        'parser_configuration': 'minicpm5 (operator-declared, not server-attested)',
        'model': args.model, 'model_revision': args.model_revision,
        'server_revision': args.server_revision,
        'template_path': str(args.template_path) if args.template_path else None,
        'template_sha256': hashlib.sha256(args.template_path.read_bytes()).hexdigest() if args.template_path else None,
        'sampling': {'temperature': args.temperature, 'top_p': args.top_p,
                     'max_tokens': args.max_tokens, 'enable_thinking': args.thinking},
        'stream': args.stream, 'repetitions': args.repetitions, 'results': [], 'complete': False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(dumps(report) + '\n')

    save()
    try:
        if not args.model:
            status, raw, response = http(args.base_url, args.api_key, '/models', args.timeout)
            report['model_discovery'] = {'status': status, 'raw_response': raw}
            if status != 200 or response is None:
                raise ValueError('model discovery failed')
            args.model = response['data'][0]['id']
            report['model'] = args.model
        for repetition in range(args.repetitions):
            for case, expected in cases():
                names = list(dict.fromkeys(name for name, _ in expected))
                tools = [{'type': 'function', 'function': {'name': name,
                    'description': 'Capture the exact supplied arguments for a simulated test.',
                    'parameters': {'type': 'object', 'additionalProperties': True}}} for name in names]
                example = '\n'.join(xml_call(name, arguments) for name, arguments in expected)
                prompt = ('This is a simulated tool formatting test. Call these tools in the given order '
                    'with exactly these argument names, JSON values and types: ' + dumps(expected) +
                    '\nOutput only the following MiniCPM5 XML format. Parameter bodies are JSON literals; '
                    'preserve JSON quotes for strings, including strings spelling numbers or null.\n' + example)
                payload = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
                    'tools': tools, 'tool_choice': 'auto', 'stream': args.stream, **report['sampling']}
                if args.stream:
                    payload['stream_options'] = {'include_usage': True}
                row = {'case': case, 'repetition': repetition, 'expected': expected, 'request': payload}
                report['results'].append(row)
                save()
                started = time.monotonic()
                try:
                    status, raw, response = http(args.base_url, args.api_key, '/chat/completions', args.timeout, payload)
                    row.update(status=status, raw_response=raw, response=response)
                    row.update(score(response, expected))
                    if case == 'tool_feedback' and row['api_shape_ok']:
                        message = response['choices'][0]['message']
                        feedback = payload | {'tool_choice': 'none', 'messages': payload['messages'] + [message] + [
                            {'role': 'tool', 'tool_call_id': call['id'], 'content': 'Simulated success. Receipt is RK3588_FEEDBACK_OK. Reply with this receipt exactly.'}
                            for call in message['tool_calls']]}
                        row['feedback_request'] = feedback
                        save()
                        code, raw, answer = http(args.base_url, args.api_key, '/chat/completions', args.timeout, feedback)
                        row['feedback'] = {'status': code, 'raw_response': raw, 'response': answer}
                        final_message = answer['choices'][0]['message'] if code == 200 and answer else {}
                        row['feedback_ok'] = bool('RK3588_FEEDBACK_OK' in (final_message.get('content') or '') and not final_message.get('tool_calls'))
                    row['passed'] = row['expected_values_ok'] and (case != 'tool_feedback' or row.get('feedback_ok', False))
                except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                    row.update(error=str(error), passed=False)
                finally:
                    row['elapsed_seconds'] = time.monotonic() - started
                    save()
        report['complete'] = True
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        report['error'] = str(error)
    finally:
        report['summary'] = {'finished': sum('passed' in row for row in report['results']),
                             'passed': sum(row.get('passed', False) for row in report['results'])}
        save()
    print(dumps(report['summary']))
    return 0 if report['complete'] and all(row.get('passed') for row in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
