"""Regression tests for clean-clone experiment orchestration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ee6483.scripts import run_experiments
from ee6483.scripts.run_experiments import already_done, expected_artifacts


class ExperimentCompletionTests(unittest.TestCase):
    def command(self, root: Path) -> list[str]:
        return [
            "python",
            "-m",
            "ee6483.src.train",
            "--exp-id",
            "guard_test",
            "--seed",
            "7",
            "--out-root",
            str(root),
        ]

    def test_metrics_without_checkpoint_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = self.command(Path(tmp))
            metrics, _ = expected_artifacts(command)
            metrics.parent.mkdir(parents=True)
            metrics.write_text(json.dumps({"best_val_acc": 0.9}), encoding="utf-8")
            self.assertFalse(already_done(command))

    def test_valid_metrics_and_nonempty_checkpoint_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = self.command(Path(tmp))
            metrics, checkpoint = expected_artifacts(command)
            metrics.parent.mkdir(parents=True)
            metrics.write_text(json.dumps({"best_val_acc": 0.9}), encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            self.assertTrue(already_done(command))

    def test_corrupt_metrics_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = self.command(Path(tmp))
            metrics, checkpoint = expected_artifacts(command)
            metrics.parent.mkdir(parents=True)
            metrics.write_text("not json", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            self.assertFalse(already_done(command))

    def test_non_training_command_is_never_skipped(self):
        command = ["python", "-m", "ee6483.scripts.eval_logit_adjust"]
        self.assertIsNone(expected_artifacts(command))
        self.assertFalse(already_done(command))

    def test_orchestrator_propagates_job_failure(self):
        fake_phase = {"failure_test": [(["python", "broken.py"], "broken")]}
        with patch.object(run_experiments, "PHASES", fake_phase), \
             patch.object(run_experiments, "job", return_value=3), \
             patch("sys.argv", ["run_experiments", "--phases", "failure_test"]):
            self.assertEqual(run_experiments.main(), 3)


if __name__ == "__main__":
    unittest.main()
