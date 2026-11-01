import unittest
from monitoring.server import parse_metrics, summarize


class MetricsTests(unittest.TestCase):
    def test_labels_aggregate_and_nonfinite_ignored(self):
        metrics = parse_metrics('# HELP x\nx{a="b"} 3\nx{a="c"} 4\ny NaN\nz +Inf')
        self.assertEqual(metrics, {'x': 7})

    def test_interval_rates_means_and_hit_ratio(self):
        previous = {'vllm:request_success_total': 10,
                    'vllm:time_to_first_token_seconds_sum': 4,
                    'vllm:time_to_first_token_seconds_count': 5,
                    'vllm:prefix_cache_hits_total': 100,
                    'vllm:prefix_cache_queries_total': 200}
        current = dict(previous)
        current.update({'vllm:request_success_total': 14,
                        'vllm:time_to_first_token_seconds_sum': 5,
                        'vllm:time_to_first_token_seconds_count': 7,
                        'vllm:prefix_cache_hits_total': 180,
                        'vllm:prefix_cache_queries_total': 300})
        result = summarize(current, previous, 2)
        self.assertEqual(result['requests_per_second'], 2)
        self.assertEqual(result['ttft_seconds'], .5)
        self.assertEqual(result['prefix_hit_rate'], .8)

    def test_missing_idle_and_counter_reset(self):
        current = {'vllm:request_success_total': 3}
        self.assertIsNone(summarize(current, {}, 2)['requests_per_second'])
        self.assertEqual(summarize(current, current, 2)['requests_per_second'], 0)
        self.assertIsNone(summarize(current, current, 2)['ttft_seconds'])
        self.assertIsNone(summarize(current, {'vllm:request_success_total': 9}, 2)['requests_per_second'])
        self.assertIsNone(summarize(current, current, 2)['prefix_hit_rate'])


if __name__ == '__main__':
    unittest.main()
