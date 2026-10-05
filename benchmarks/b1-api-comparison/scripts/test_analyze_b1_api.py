import copy
import unittest
from analyze_b1_api import compare


def sample(scale=1.0, sessions=8):
    return {
        s: {
            f"{c}-{i}": dict(
                category=c, ttft_s=scale * (1 + s / 100), decode_tps=100 / scale
            )
            for c in ("code", "knowledge")
            for i in range(3)
        }
        for s in range(sessions)
    }


class Gates(unittest.TestCase):
    def test_direction(self):
        self.assertTrue(compare(sample(0.9), sample(), "ttft_s", 100)["better"])
        self.assertTrue(compare(sample(0.9), sample(), "decode_tps", 100)["better"])
        self.assertFalse(compare(sample(1.1), sample(), "ttft_s", 100)["no_worse"])

    def test_not_enough_replicates(self):
        self.assertFalse(compare(sample(0.5, 1), sample(1, 1), "ttft_s", 100)["better"])

    def test_category_regression_is_not_hidden(self):
        candidate = copy.deepcopy(sample(0.5))
        for rows in candidate.values():
            for r in rows.values():
                if r["category"] == "knowledge":
                    r["ttft_s"] *= 3
        self.assertFalse(compare(candidate, sample(), "ttft_s", 100)["better"])

    def test_exact_equality_is_not_superiority(self):
        result = compare(sample(), sample(), "ttft_s", 100)
        self.assertTrue(result["no_worse"])
        self.assertFalse(result["better"])


if __name__ == "__main__":
    unittest.main()
