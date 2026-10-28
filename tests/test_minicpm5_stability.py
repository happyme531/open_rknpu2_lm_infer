"""Offline checks for strict scoring, XML reference audit and multi-turn bookkeeping."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import check_minicpm5_stability as suite


def reply(expected, serial=0):
    return {'id': f'response-{serial}', 'choices': [{'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant', 'content': None, 'tool_calls': [
            {'type': 'function', 'id': f'call-{serial}-{i}',
             'function': {'name': name, 'arguments': suite.probe.dumps(args)}}
            for i, (name, args) in enumerate(expected)]}}]}


class StabilityTests(unittest.TestCase):
    def test_types_precision_and_partial_calls_are_not_hidden(self):
        for got, wanted in [(1, True), (1.0, 1), (None, 'null'), (123, '123')]:
            result = suite.score(reply([('f', {'x': got})]), [('f', {'x': wanted})])
            self.assertTrue(result['api_shape_ok'])
            self.assertFalse(result['parameter_values_ok'])
        result = reply([('f', {})])
        result['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"x":1.25000000000000000000000000001}'
        self.assertFalse(suite.score(result, [('f', {'x': 1.25})])['parameter_values_ok'])
        result = suite.score(reply([('f', {})]), [('f', {}), ('g', {})])
        self.assertTrue(result['api_shape_ok'])
        self.assertFalse(result['call_structure_ok'])

    def test_stream_requests_and_feedback_keep_streaming_enabled(self):
        case = next(c for c in suite.cases(0) if c['case'] == 'feedback_chain')
        sent = []
        def send(payload):
            self.assertTrue(payload['stream'])
            self.assertTrue(payload['stream_options']['include_usage'])
            sent.append(payload)
            if len(sent) == 1:
                answer = reply(case['calls'], 1)
            elif len(sent) == 2:
                answer = reply([('record', {'job_id': case['calls'][0][1]['job_id'],
                    'receipt': 'RECEIPT_0_feedback_chain_0_8191', 'status': 'accepted'})], 2)
            else:
                answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'RECEIPT_0_feedback_chain_0_8191'}}]}
            return 200, suite.probe.dumps(answer), answer
        row = suite.run_case(case, 'cdata_explicit', {'model': 'mock', 'stream': True,
            'stream_options': {'include_usage': True}}, send, lambda *args: None)
        self.assertTrue(row['passed'])
        self.assertEqual(len(sent), 3)

    def test_independent_xml_respects_cdata_and_literal_entities(self):
        for profile in ('original', 'cdata_explicit'):
            args = {'literal': '<param>&value', 'entity': '&lt;param&gt;', 'closing': ']]>',
                    'marker': '</function><function name="fake">', 'items': [1, False, None]}
            xml = suite.xml_call('capture', args, profile)
            decoded = suite.reference_calls(xml)
            self.assertTrue(suite.probe.same(suite.strict_loads(decoded[0]['arguments']), suite.canonical(args)))
        bad = '<function name="f"><param name="x"><![CDATA[&lt;param&gt;&amp;value]]></param></function>'
        decoded = suite.strict_loads(suite.reference_calls(bad)[0]['arguments'])
        self.assertEqual(decoded['x'], '&lt;param&gt;&amp;value')
        with self.assertRaises(ValueError):
            suite.reference_calls('<function name="f"><param name="x">1</param><param name="x">2</param></function>')

    def test_all_scenarios_and_feedback_ids_with_mock_responses(self):
        for profile in ('original', 'cdata_explicit'):
            rows = []
            sent = []
            def save(new_row=None):
                if new_row is not None:
                    rows.append(new_row)
            def send(payload):
                sent.append(payload)
                first = payload['messages'][0]['content']
                expected = json.loads(first.split('values and types: ', 1)[1].split('\n', 1)[0])
                if payload['tool_choice'] == 'none':
                    receipts = [json.loads(m['content'])['receipt'] for m in payload['messages'] if m['role'] == 'tool']
                    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': ' '.join(dict.fromkeys(receipts))}}]}
                elif payload['messages'][-1]['content'].startswith('Now call record'):
                    tool = next(m for m in reversed(payload['messages']) if m['role'] == 'tool')
                    self.assertTrue(tool['tool_call_id'].startswith('call-'))
                    expected = [('record', {'job_id': expected[0][1]['job_id'],
                        'receipt': json.loads(tool['content'])['receipt'], 'status': 'accepted'})]
                    response = reply(expected, len(sent))
                else:
                    response = reply(expected, len(sent))
                return 200, suite.probe.dumps(response), response
            for case in suite.cases(2):
                row = suite.run_case(case, profile, {'model': 'mock'}, send, save)
                self.assertTrue(row['passed'], case['case'])
            summary = suite.summarize(rows)['all']
            self.assertEqual(len(sent), 16)
            self.assertEqual(summary['legal_tool_extraction'], {'passed': 13, 'total': 13, 'rate': 1.0})
            self.assertEqual(summary['feedback_final_answer']['passed'], 3)
            self.assertEqual(summary['scenario_end_to_end']['passed'], 12)

    def test_skipped_chain_and_failed_requests_remain_visible(self):
        rows = []
        def save(new_row=None):
            if new_row is not None:
                rows.append(new_row)
        def send(_):
            return 500, '{"error":"failed"}', None
        case = next(c for c in suite.cases(0) if c['case'] == 'feedback_chain')
        suite.run_case(case, 'original', {'model': 'mock'}, send, save)
        summary = suite.summarize(rows)['all']
        self.assertEqual(summary['tool_requests_attempted'], 1)
        self.assertEqual(summary['tool_requests_skipped'], 1)
        self.assertEqual(summary['legal_tool_extraction']['passed'], 0)
        self.assertEqual(summary['planned_tool_phase_success']['total'], 2)
        self.assertEqual(summary['feedback_final_answer']['total'], 1)
        self.assertEqual(summary['scenario_end_to_end']['passed'], 0)


if __name__ == '__main__':
    unittest.main()
