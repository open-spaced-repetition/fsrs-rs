import importlib.util
import math
from pathlib import Path
import sys
import unittest


RUN_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("autoresearch_run", RUN_PATH)
assert SPEC is not None and SPEC.loader is not None
autoresearch_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = autoresearch_run
SPEC.loader.exec_module(autoresearch_run)


def golden(
    *,
    parameters=None,
    dataset_len=10,
    dataset_checksum=12345,
    evaluate_log_loss=0.2,
    inference_retrievability=0.9,
    trained_log_loss=0.18,
    time_series_log_loss=0.22,
):
    if parameters is None:
        parameters = [float(i) for i in range(21)]
    return {
        "dataset": {"len": dataset_len, "checksum": dataset_checksum},
        "evaluate": {"log_loss": evaluate_log_loss, "rmse_bins": 0.03},
        "compute_parameters": parameters,
        "trained_evaluate": {"log_loss": trained_log_loss, "rmse_bins": 0.02},
        "time_series_evaluate": {
            "log_loss": time_series_log_loss,
            "rmse_bins": 0.04,
        },
        "inference": {
            "current_retrievability": inference_retrievability,
            "memory_state": {"stability": 2.0, "difficulty": 6.0},
            "memory_state_batch": [
                {"stability": 2.0, "difficulty": 6.0},
                {"stability": 3.0, "difficulty": 7.0},
            ],
            "next_states": {
                "again": {
                    "memory": {"stability": 1.0, "difficulty": 8.0},
                    "interval": 1.0,
                },
                "hard": {
                    "memory": {"stability": 2.0, "difficulty": 7.0},
                    "interval": 2.0,
                },
                "good": {
                    "memory": {"stability": 3.0, "difficulty": 6.0},
                    "interval": 3.0,
                },
                "easy": {
                    "memory": {"stability": 4.0, "difficulty": 5.0},
                    "interval": 4.0,
                },
            },
        },
    }


def compare(expected, actual, mode="logloss-band", band=0.001):
    return autoresearch_run.compare_golden_for_mode(
        expected,
        actual,
        mode=mode,
        rel_tol=1e-4,
        abs_tol=1e-4,
        logloss_band=band,
    )


class CompareGoldenForModeTest(unittest.TestCase):
    def test_bit_exact_passes_identical_golden(self):
        baseline = golden()

        result = compare(baseline, golden(), mode="bit-exact")

        self.assertTrue(result["ok"])
        self.assertEqual(result["mismatch_count"], 0)

    def test_bit_exact_rejects_parameter_drift(self):
        baseline = golden()
        candidate = golden(parameters=[float(i) + 0.01 for i in range(21)])

        result = compare(baseline, candidate, mode="bit-exact")

        self.assertFalse(result["ok"])
        self.assertGreater(result["mismatch_count"], 0)

    def test_logloss_band_allows_parameter_drift_inside_band(self):
        baseline = golden()
        candidate = golden(
            parameters=[float(i) + 0.5 for i in range(21)],
            trained_log_loss=0.1809,
            time_series_log_loss=0.2192,
        )

        result = compare(baseline, candidate)

        self.assertTrue(result["ok"])
        self.assertEqual(result["mismatch_count"], 0)
        self.assertEqual(result["parameter_diagnostics"]["drift_count"], 21)

    def test_logloss_band_rejects_dataset_change(self):
        baseline = golden()
        candidate = golden(dataset_checksum=999)

        result = compare(baseline, candidate)

        self.assertFalse(result["ok"])
        self.assertIn("$.dataset.checksum", result["mismatches"][0]["path"])

    def test_logloss_band_rejects_default_evaluate_change(self):
        baseline = golden()
        candidate = golden(evaluate_log_loss=0.201)

        result = compare(baseline, candidate)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(mismatch["path"] == "$.evaluate.log_loss" for mismatch in result["mismatches"])
        )

    def test_logloss_band_rejects_inference_change(self):
        baseline = golden()
        candidate = golden(inference_retrievability=0.8)

        result = compare(baseline, candidate)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                mismatch["path"] == "$.inference.current_retrievability"
                for mismatch in result["mismatches"]
            )
        )

    def test_logloss_band_rejects_trained_logloss_over_band(self):
        baseline = golden()
        candidate = golden(trained_log_loss=0.1811)

        result = compare(baseline, candidate)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                mismatch["path"] == "$.trained_evaluate.log_loss"
                for mismatch in result["mismatches"]
            )
        )

    def test_logloss_band_rejects_time_series_logloss_over_band(self):
        baseline = golden()
        candidate = golden(time_series_log_loss=0.2211)

        result = compare(baseline, candidate)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                mismatch["path"] == "$.time_series_evaluate.log_loss"
                for mismatch in result["mismatches"]
            )
        )

    def test_logloss_band_rejects_wrong_parameter_length(self):
        baseline = golden()
        candidate = golden(parameters=[0.0] * 20)

        result = compare(baseline, candidate)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                mismatch["path"] == "$.compute_parameters.length"
                for mismatch in result["mismatches"]
            )
        )

    def test_logloss_band_rejects_nonfinite_parameter(self):
        baseline = golden()
        params = [float(i) for i in range(21)]
        params[3] = math.nan
        candidate = golden(parameters=params)

        result = compare(baseline, candidate)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                mismatch["path"] == "$.compute_parameters"
                for mismatch in result["mismatches"]
            )
        )


if __name__ == "__main__":
    unittest.main()
