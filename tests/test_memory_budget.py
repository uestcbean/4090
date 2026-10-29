import unittest
from scripts.memory_budget import GIB, estimate


class MemoryBudgetTests(unittest.TestCase):
    config = {"num_hidden_layers": 36, "num_key_value_heads": 8,
              "head_dim": 128, "parameter_count_estimate": 8_000_000_000}

    def test_hand_calculated_qwen_example(self):
        result = estimate(self.config, [8192] * 4)
        self.assertEqual(result["kv_bytes_per_token"], 147456)
        self.assertEqual(result["kv_gib"], 4.5)
        self.assertEqual(result["weights_gib_estimate"], 16_000_000_000 / GIB)

    def test_unequal_lengths_and_zero_cache(self):
        self.assertEqual(estimate(self.config, [4096, 8192])["kv_gib"], 1.6875)
        self.assertEqual(estimate(self.config, [0])["kv_gib"], 0)

    def test_weight_quantization_does_not_quantize_kv(self):
        bf16 = estimate(self.config, [8192])
        int4 = estimate(self.config, [8192], weight_bits=4)
        self.assertEqual(int4["kv_gib"], bf16["kv_gib"])
        self.assertEqual(int4["weights_gib_estimate"] * 4, bf16["weights_gib_estimate"])

    def test_insufficient_capacity(self):
        self.assertFalse(estimate(self.config, [8192] * 8)["fits_arithmetic_budget_only"])

    def test_invalid_inputs(self):
        for lengths in ([], [-1], [1.5]):
            with self.assertRaises(ValueError):
                estimate(self.config, lengths)
        for kwargs in ({"reserve_gib": -1}, {"kv_bytes": 0}, {"capacity_gib": float("nan")}):
            with self.assertRaises(ValueError):
                estimate(self.config, [8192], **kwargs)


if __name__ == "__main__":
    unittest.main()
