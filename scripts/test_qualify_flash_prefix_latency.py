import importlib.util
import json
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('prefix_latency', Path(__file__).with_name('qualify-flash-prefix-latency.py'))
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def event(value):
    return ('data: ' + json.dumps(value) + '\n').encode()


class PrefixLatencyTests(unittest.TestCase):
    def test_content_gaps_exclude_role_and_terminal_usage(self):
        lines = [event({'choices': [{'delta': {'role': 'assistant'}}]}),
                 event({'choices': [{'delta': {'content': '1'}}]}),
                 event({'choices': [{'delta': {'content': ' 2'}}]}),
                 event({'choices': [{'delta': {}, 'finish_reason': 'length'}]}),
                 event({'choices': [], 'usage': {'completion_tokens': 3}}),
                 b'data: [DONE]\n']
        clock = iter([11., 16., 17.])
        row = MOD.decode_events(lines, 10., lambda: next(clock))
        self.assertEqual(row['content'], '1 2')
        self.assertEqual(row['event_seconds'], [1., 6.])
        self.assertEqual(row['max_content_gap_s'], 5.)
        self.assertEqual(row['ttft_s'], 1.)
        self.assertEqual(row['decode_s'], 6.)
        self.assertEqual(row['total_s'], 7.)

    def test_missing_terminal_events_is_failure(self):
        with self.assertRaisesRegex(ValueError, 'incomplete SSE'):
            MOD.decode_events([event({'choices': [{'delta': {'content': 'x'}}]})], 0., lambda: 1.)

    def test_summary_keeps_outliers_without_small_sample_p95(self):
        rows = [dict(ttft_s=x, decode_s=x, max_content_gap_s=x) for x in [1., 1., 21.]]
        data = MOD.summary(rows)
        self.assertEqual(data['decode_s'], dict(median=1., minimum=1., maximum=21.))
        self.assertNotIn('p95_nearest_rank', data['decode_s'])
        rows = [dict(ttft_s=x, decode_s=x, max_content_gap_s=x) for x in range(1, 21)]
        self.assertEqual(MOD.summary(rows)['decode_s']['p95_nearest_rank'], 19)

    def test_process_deltas_keep_residency_and_io_distinct(self):
        before = {key: 10 for key in MOD.RUSAGE_FIELDS}
        after = {key: 20 for key in MOD.RUSAGE_FIELDS}
        self.assertEqual(MOD.counter_delta(before, after)['pageins'], 10)
        self.assertEqual(MOD.counter_delta(before, after)['diskio_bytesread'], 10)
        self.assertNotIn('wired_size', MOD.counter_delta(before, after))
        self.assertIsNone(MOD.counter_delta(before, {'error': 3}))


if __name__ == '__main__':
    unittest.main()
