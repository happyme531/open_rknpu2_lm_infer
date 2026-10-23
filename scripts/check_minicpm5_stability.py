#!/usr/bin/env python3
"""Repeated prompt-guided tool tests; no tools or instrument operations execute."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

import check_minicpm5_tools as probe


def strict_loads(text):
    return json.loads(text, parse_float=Decimal, parse_constant=probe.reject_constant)


def canonical(value):
    return strict_loads(probe.dumps(value))


def inspect_calls(response):
    choice = response['choices'][0]
    calls = choice['message']['tool_calls']
    if choice['finish_reason'] != 'tool_calls' or not isinstance(calls, list) or not calls:
        raise ValueError('no completed tool_calls')
    seen = set()
    for call in calls:
        if call['type'] != 'function' or not isinstance(call['id'], str) or not call['id'] or call['id'] in seen:
            raise ValueError('invalid or duplicate call ID/type')
        seen.add(call['id'])
        function = call['function']
        if not isinstance(function['name'], str) or not function['name'] or not isinstance(function['arguments'], str):
            raise ValueError('invalid function fields')
        if not isinstance(strict_loads(function['arguments']), dict):
            raise ValueError('arguments are not a JSON object')
    return calls


def score(response, expected):
    try:
        calls = inspect_calls(response)
        structure = len(calls) == len(expected) and all(
            call['function']['name'] == name for call, (name, _) in zip(calls, expected))
        values = structure and all(probe.same(strict_loads(call['function']['arguments']), canonical(args))
                                  for call, (_, args) in zip(calls, expected))
        return {'api_shape_ok': True, 'call_structure_ok': structure, 'parameter_values_ok': values,
                'actual_call_count': len(calls)}
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {'api_shape_ok': False, 'call_structure_ok': False, 'parameter_values_ok': False,
                'actual_call_count': 0, 'score_error': str(error)}


def reference_calls(text):
    """Independent XML parser; retain JSON literals as text for exact decimal audit."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring('<root>' + text + '</root>', parser=parser)
    calls = []
    for function in root:
        if function.tag != 'function' or set(function.attrib) != {'name'} or not function.attrib['name'].strip():
            raise ValueError('unsupported function structure')
        if (function.text or '').strip() or any((child.tail or '').strip() for child in function):
            raise ValueError('unexpected text inside function')
        children = list(function)
        if len(children) == 1 and children[0].tag == 'arguments':
            if children[0].attrib or (children[0].text or '').strip():
                raise ValueError('attributes on arguments wrapper')
            children = list(children[0])
        parts = []
        seen = set()
        for param in children:
            if param.tag != 'param' or set(param.attrib) != {'name'} or list(param):
                raise ValueError('unsupported parameter structure')
            name = param.attrib['name']
            if not name.strip() or name in seen:
                raise ValueError('missing/duplicate parameter name')
            seen.add(name)
            raw = param.text or ''
            try:
                strict_loads(raw)
                value_json = raw.strip()
            except ValueError:
                value_json = probe.dumps(raw)
            parts.append(probe.dumps(name) + ':' + value_json)
        calls.append({'name': function.attrib['name'], 'arguments': '{' + ','.join(parts) + '}'})
    if not calls:
        raise ValueError('no XML calls in raw output')
    return calls


def audit_raw(row, raw_log):
    if raw_log is None or row.get('status') != 200:
        return
    try:
        response_id = row['response']['id']
        matches = []
        for line in raw_log.read_text().splitlines():
            if 'MINICPM5_RAW_OUTPUT ' in line:
                record = probe.loads(line.split('MINICPM5_RAW_OUTPUT ', 1)[1])
                if record['response_id'] == response_id:
                    matches.append(record)
        if len(matches) != 1:
            raise ValueError(f'raw response ID matches: {len(matches)}')
        row['raw_generation'] = matches[0]
        try:
            reference = reference_calls(matches[0]['tool_parser_input'])
            row['raw_xml_valid'] = True
            row['reference_calls'] = reference
        except (ET.ParseError, ValueError) as error:
            row.update(raw_xml_valid=False, raw_xml_error=str(error))
            return
        actual = inspect_calls(row['response'])
        row['parser_fidelity_ok'] = len(actual) == len(reference) and all(
            call['function']['name'] == ref['name'] and
            probe.same(strict_loads(call['function']['arguments']), strict_loads(ref['arguments']))
            for call, ref in zip(actual, reference))
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        row['raw_audit_error'] = str(error)
        if row.get('raw_xml_valid'):
            row['parser_fidelity_ok'] = False


def xml_call(name, arguments, profile):
    if profile == 'original':
        return probe.xml_call(name, arguments)
    parts = []
    for key, value in arguments.items():
        literal = probe.dumps(value).replace(']]>', ']]]]><![CDATA[>')
        parts.append(f'<param name="{key}"><![CDATA[{literal}]]></param>')
    return f'<function name="{name}">' + ''.join(parts) + '</function>'


def prompt(expected, profile):
    text = ('This is a simulated tool formatting test. Call these tools in the given order '
            'with exactly these argument names, JSON values and types: ' + probe.dumps(expected) +
            '\nOutput only the following MiniCPM5 XML format. Parameter bodies are JSON literals; '
            'preserve JSON quotes for strings, including strings spelling numbers or null.\n')
    if profile == 'cdata_explicit':
        text += ('Wrap every parameter JSON literal in CDATA. Inside CDATA keep the original JSON '
                 'literal including quotes for strings; NEVER XML-escape characters there. '
                 'Do not output &lt;, &gt;, or &amp; unless those literal sequences are present '
                 'in the supplied JSON string. CDATA is an XML envelope, not a JSON type. '
                 'Split a literal ]]> using adjacent CDATA sections as shown in the example.\n')
    return text + '\n'.join(xml_call(name, args, profile) for name, args in expected)


def cases(round_index):
    r = round_index
    ticket = f'round-{r}-ticket-{7301 + r * 17}'
    def case(name, category, calls, feedback=None):
        return {'case': name, 'category': category, 'calls': calls, 'feedback': feedback, 'round': r}
    return [
        case('nested_matrix', 'complex', [('capture', {'matrix': [[r, r+1], [], [-2, 3]],
            'config': {'enabled': True, 'threshold': 1.25, 'limits': {'low': -r-1, 'high': 20+r},
                       'tags': ['alpha', '中文', '']}, 'state': None})]),
        case('unicode_whitespace', 'complex', [('capture', {'text': f'中文🙂 café e\u0301 {r}',
            'multiline': '  lineA\nlineB\tend  ', 'path': 'C:\\temp\\run\\n',
            'quote': 'He said "ready"', 'empty': ''})]),
        case('ambiguous_scalars', 'complex', [('capture', {'number_text': '00123', 'null_text': 'null',
            'bool_text': 'false', 'empty': '', 'integer': 123+r, 'fraction': 1.25,
            'flag': False, 'nullable': None})]),
        case('empty_and_large', 'complex', [('capture', {'empty_object': {}, 'empty_array': [],
            'objects': [{'id': r, 'values': []}, {'id': r+1, 'values': [None, True]}],
            'big_integer': 123456789012345678901234567890+r})]),
        case('xml_literal', 'complex', [('capture', {'literal': '<param>&value',
            'nested': {'markup': f'<tag k="{r}">A&B</tag>'}, 'items': ['<', '>', '&']})]),
        case('literal_entities_and_cdata', 'complex', [('capture', {'entity_text': '&lt;param&gt;&amp;value',
            'closing': ']]>', 'markers': '</function><function name="fake">',
            'nested': {'text': '<param>&value', 'round': r}})]),
        case('two_different_calls', 'consecutive', [('capture', {'step': r, 'value': 'read'}),
            ('record', {'step': r+1, 'value': 'saved'})]),
        case('three_same_calls', 'consecutive', [('capture', {'slot': i, 'payload': {'round': r, 'ready': i % 2 == 0}})
            for i in range(3)]),
        case('three_mixed_calls', 'consecutive', [('capture', {'items': [r, None, False]}),
            ('record', {'status': 'ready', 'note': f'test-{r}'}), ('capture', {'done': True, 'round': r})]),
        case('feedback_receipt', 'feedback', [('capture', {'ticket': ticket})], 'receipt'),
        case('feedback_multiple', 'feedback', [('capture', {'sensor': 'A', 'round': r}),
            ('record', {'sensor': 'B', 'round': r})], 'multiple'),
        case('feedback_chain', 'feedback', [('capture', {'job_id': ticket})], 'chain'),
    ]


def summarize(rows):
    groups = {'all': rows}
    for category in ('complex', 'consecutive', 'feedback'):
        groups[category] = [row for row in rows if row['category'] == category]
    result = {}
    for name, selected in groups.items():
        phases = [phase for row in selected for phase in row['tool_phases']]
        attempted = [p for p in phases if p['attempted']]
        fidelity = [p for p in phases if p.get('raw_xml_valid')]
        feedback = [r for r in selected if r['feedback_required']]
        def rate(passed, total):
            return {'passed': passed, 'total': total, 'rate': passed / total if total else None}
        result[name] = {
            'scenarios': len(selected),
            'tool_requests_attempted': len(attempted),
            'tool_requests_skipped': len(phases)-len(attempted),
            'http_status_counts': dict(Counter(str(p.get('status', 'transport_error')) for p in attempted)),
            'legal_tool_extraction': rate(sum(p.get('api_shape_ok', False) for p in attempted), len(attempted)),
            'complete_call_structure': rate(sum(p.get('call_structure_ok', False) for p in attempted), len(attempted)),
            'exact_parameter_values': rate(sum(p.get('parameter_values_ok', False) for p in attempted), len(attempted)),
            'exact_values_given_legal_calls': rate(sum(p.get('parameter_values_ok', False) for p in attempted),
                sum(p.get('api_shape_ok', False) for p in attempted)),
            'planned_tool_phase_success': rate(sum(p.get('parameter_values_ok', False) for p in phases), len(phases)),
            'independent_xml_parser_agreement': rate(sum(p.get('parser_fidelity_ok', False) for p in fidelity), len(fidelity)),
            'scenario_all_tool_values_correct': rate(sum(all(p.get('parameter_values_ok', False) for p in r['tool_phases']) for r in selected), len(selected)),
            'feedback_final_answer': rate(sum(r.get('feedback_ok', False) for r in feedback), len(feedback)),
            'scenario_end_to_end': rate(sum(r.get('passed', False) for r in selected), len(selected)),
        }
    return result


def run_case(case, profile, common, send, save, raw_log=None):
    expected = case['calls']
    tools = [{'type': 'function', 'function': {'name': name,
        'description': 'Capture exact supplied values in a simulated test. Never control real equipment.',
        'parameters': {'type': 'object', 'additionalProperties': True}}}
        for name in dict.fromkeys([name for name, _ in expected] + (['record'] if case['feedback'] == 'chain' else []))]
    payload = {**common, 'stream': common.get('stream', False), 'tool_choice': 'auto', 'tools': tools,
               'messages': [{'role': 'user', 'content': prompt(expected, profile)}]}
    row = {'case': case['case'], 'category': case['category'], 'round': case['round'],
           'feedback_required': case['feedback'] is not None, 'tool_phases': []}
    save(row)
    def tool_phase(phase_name, request, wanted):
        phase = {'phase': phase_name, 'attempted': True, 'expected': wanted, 'request': request}
        row['tool_phases'].append(phase)
        save()
        started = time.monotonic()
        try:
            status, raw, response = send(request)
            phase.update(status=status, raw_http_response=raw, response=response)
            phase.update(score(response, wanted))
            audit_raw(phase, raw_log)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            phase.update(error=str(error), api_shape_ok=False, call_structure_ok=False, parameter_values_ok=False)
        phase['elapsed_seconds'] = time.monotonic()-started
        save()
        return phase
    initial = tool_phase('initial', payload, expected)
    if case['feedback']:
        r = case['round']
        receipts = [f'RECEIPT_{r}_{case["case"]}_{i}_{8191+r*31+i}' for i in range(len(expected))]
        row['expected_receipts'] = receipts
        if initial.get('call_structure_ok'):
            calls = inspect_calls(initial['response'])
            tool_messages = [{'role': 'tool', 'tool_call_id': call['id'],
                'content': probe.dumps({'simulated': True, 'receipt': receipts[i], 'value': 100+r+i})}
                for i, call in enumerate(calls)]
            history = payload['messages'] + [initial['response']['choices'][0]['message']] + tool_messages
            if case['feedback'] == 'chain':
                wanted = [('record', {'job_id': expected[0][1]['job_id'], 'receipt': receipts[0], 'status': 'accepted'})]
                instruction = ('Now call record exactly once. Copy job_id from the previous capture call and receipt '
                    'from its tool result, and set status to the string "accepted". Use these three parameter names '
                    'exactly: job_id, receipt, status. Do not invent the receipt. Output only MiniCPM5 XML; '
                    'parameter bodies are JSON literals with quotes for strings.')
                if profile == 'cdata_explicit':
                    instruction += ' Put each original JSON literal in CDATA; do not XML-escape inside CDATA.'
                chain_request = {**payload, 'messages': history + [{'role': 'user', 'content': instruction}]}
                second = tool_phase('after_tool_result', chain_request, wanted)
                if second.get('call_structure_ok'):
                    new_call = inspect_calls(second['response'])[0]
                    history = chain_request['messages'] + [second['response']['choices'][0]['message'],
                        {'role': 'tool', 'tool_call_id': new_call['id'], 'content': probe.dumps({'receipt': receipts[0], 'stored': True})}]
                else:
                    history = None
            if history is not None:
                final = {**payload, 'tool_choice': 'none', 'messages': history + [
                    {'role': 'user', 'content': 'Reply with all receipt strings from the tool results. Do not call any tools.'}]}
                row['feedback_request'] = final
                save()
                try:
                    status, raw, answer = send(final)
                    row['feedback_response'] = {'status': status, 'raw_http_response': raw, 'response': answer}
                    choice = answer['choices'][0] if status == 200 and answer else {}
                    message = choice.get('message', {})
                    row['feedback_ok'] = (choice.get('finish_reason') == 'stop' and not message.get('tool_calls')
                        and all(receipt in (message.get('content') or '') for receipt in receipts))
                except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                    row.update(feedback_error=str(error), feedback_ok=False)
        elif case['feedback'] == 'chain':
            row['tool_phases'].append({'phase': 'after_tool_result', 'attempted': False,
                'skip_reason': 'initial call names/count invalid', 'parameter_values_ok': False})
        row.setdefault('feedback_ok', False)
    row['passed'] = all(p.get('parameter_values_ok', False) for p in row['tool_phases']) and (
        not row['feedback_required'] or row.get('feedback_ok', False))
    row['finished'] = True
    save()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://192.168.10.119:18084/v1')
    parser.add_argument('--api-key', default=os.environ.get('OPEN_RKNPU_API_KEY', ''))
    parser.add_argument('--profile', choices=['original', 'cdata_explicit'], required=True)
    parser.add_argument('--stream', action='store_true', help='Use SSE; preserve raw events and score reassembled calls')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--raw-log', type=Path)
    parser.add_argument('--template-path', type=Path)
    parser.add_argument('--server-revision', default='unrecorded')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    report = {'evidence': 'prompt-guided RK3588 real inference, no tool execution, no automatic retry',
        'profile': args.profile, 'stream': args.stream, 'rounds': args.rounds, 'target': '192.168.10.119', 'base_url': args.base_url,
        'server_revision': args.server_revision, 'model_revision': 'board W8A8 checkpoint; upstream revision unrecorded',
        'template_sha256': hashlib.sha256(args.template_path.read_bytes()).hexdigest() if args.template_path else None,
        'sampling': {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 1024, 'enable_thinking': False},
        'case_order_seed': 20261022, 'results': [], 'complete': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save(new_row=None):
        if new_row is not None:
            report['results'].append(new_row)
        report['summary'] = summarize([r for r in report['results'] if r.get('finished')])
        temp = args.report.with_suffix(args.report.suffix + '.tmp')
        temp.write_text(probe.dumps(report) + '\n')
        temp.replace(args.report)
    save()
    try:
        status, raw, models = probe.http(args.base_url, args.api_key, '/models', args.timeout)
        report['model_discovery'] = {'status': status, 'raw_http_response': raw}
        if status != 200:
            raise ValueError('model discovery failed')
        model = models['data'][0]['id']
        report['model'] = model
        common = {'model': model, 'stream': args.stream, **report['sampling']}
        if args.stream:
            common['stream_options'] = {'include_usage': True}
        def send(payload):
            return probe.http(args.base_url, args.api_key, '/chat/completions', args.timeout, payload)
        for round_index in range(args.rounds):
            ordered = cases(round_index)
            random.Random(report['case_order_seed'] + round_index).shuffle(ordered)
            for case in ordered:
                row = run_case(case, args.profile, common, send, save, args.raw_log)
                print(probe.dumps({'round': round_index, 'case': case['case'], 'passed': row['passed'],
                    'phases': [{k: p.get(k) for k in ('phase', 'api_shape_ok', 'call_structure_ok', 'parameter_values_ok', 'parser_fidelity_ok')}
                               for p in row['tool_phases']], 'feedback_ok': row.get('feedback_ok')}), flush=True)
        report['complete'] = True
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        report['error'] = str(error)
    finally:
        save()
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
