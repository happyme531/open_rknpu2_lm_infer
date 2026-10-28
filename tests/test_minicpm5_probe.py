"""Offline tests of the board probe's scoring and report lifecycle."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'scripts/check_minicpm5_tools.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def response(expected):
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
        'content': None, 'tool_calls': [
            {'id': f'call_{i}', 'type': 'function', 'function': {'name': name, 'arguments': probe.dumps(args)}}
            for i, (name, args) in enumerate(expected)]}}]}


class ProbeTests(unittest.TestCase):
    def test_type_mismatches_do_not_pass(self):
        for got, want in [(1, True), (1.0, 1), (None, 'null'), (123, '123')]:
            scored = probe.score(response([('f', {'x': got})]), [('f', {'x': want})])
            self.assertTrue(scored['api_shape_ok'])
            self.assertFalse(scored['expected_values_ok'])

    def test_invalid_api_is_distinct_from_bad_model_values(self):
        scored = probe.score({'error': 'server failed'}, [('f', {})])
        self.assertFalse(scored['api_shape_ok'])
        self.assertFalse(scored['expected_values_ok'])
        bad = response([('f', {})])
        bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"x":NaN}'
        self.assertFalse(probe.score(bad, [('f', {})])['api_shape_ok'])

    def test_guided_requests_and_feedback_report(self):
        pending = iter(probe.cases())
        def fake_http(base, key, endpoint, timeout, payload=None):
            if endpoint == '/models':
                answer = {'data': [{'id': 'mock-model'}]}
            elif payload['tool_choice'] == 'none':
                self.assertEqual(payload['messages'][-1]['tool_call_id'], 'call_0')
                answer = {'choices': [{'message': {'content': 'RK3588_FEEDBACK_OK'}}]}
            else:
                _, expected = next(pending)
                self.assertFalse(payload['stream'])
                for name, args in expected:
                    self.assertIn(probe.xml_call(name, args), payload['messages'][0]['content'])
                answer = response(expected)
            return 200, probe.dumps(answer), answer
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            with patch.object(probe, 'http', side_effect=fake_http), patch.object(sys, 'argv', ['probe', '--report', str(output)]):
                self.assertEqual(probe.main(), 0)
            report = json.loads(output.read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['summary'], {'finished': 5, 'passed': 5})
            self.assertNotIn('api_key', report)

    def test_sse_reassembles_fragmented_calls_and_usage(self):
        chunks = [
            {'choices': [{'index': 0, 'delta': {'content': 'before', 'tool_calls': [
                {'index': 0, 'id': 'call_0', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"x":'}}]}, 'finish_reason': None}]},
            {'choices': [{'index': 0, 'delta': {'tool_calls': [
                {'index': 0, 'function': {'arguments': '123}'}},
                {'index': 1, 'id': 'call_1', 'type': 'function', 'function': {'name': 'g', 'arguments': '{}'}}]}, 'finish_reason': None}]},
            {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'total_tokens': 5}},
        ]
        raw = ''.join('data: ' + probe.dumps(chunk) + '\r\n\r\n' for chunk in chunks) + 'data: [DONE]\r\n\r\n'
        answer = probe.assemble_sse(raw)
        self.assertTrue(probe.score(answer, [('f', {'x': 123}), ('g', {})])['expected_values_ok'])
        self.assertEqual(answer['choices'][0]['message']['content'], 'before')
        self.assertEqual(answer['usage'], {'total_tokens': 5})
        with self.assertRaises(ValueError):
            probe.assemble_sse(raw.replace('data: [DONE]', ': not complete'))
        with self.assertRaises(ValueError):
            probe.assemble_sse('data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n')

    def test_discovery_failure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            with patch.object(probe, 'http', side_effect=OSError('unreachable')), patch.object(sys, 'argv', ['probe', '--report', str(output)]):
                self.assertEqual(probe.main(), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['error'], 'unreachable')


if __name__ == '__main__':
    unittest.main()
