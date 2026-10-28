"""Resume orchestration checks using synthetic epoch records, without training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from playground import run_ml1m_benchmark as benchmark


def epoch_row(epoch):
    return {"epoch": epoch, "metrics": {metric: .1 for metric in benchmark.METRICS + benchmark.EXTRA_METRICS},
            "elapsed_seconds": epoch + 1, "peak_memory_bytes": 1}


class ResumeTests(unittest.TestCase):
    def test_paper_check_accepts_close_baselines_and_positive_mol_gains(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {"paper_verification": {"required": True, "absolute_tolerance": .01}}
            for name in ("sasrec", "sasrec_mol"):
                metrics = dict(zip(benchmark.METRICS, benchmark.TARGETS[name]))
                metrics["hr@10"] -= .003
                benchmark.verify_paper(run, name, metrics, manifest)
            policy = json.loads((run / "paper_consistency.json").read_text())
            self.assertTrue(all(row["passed"] for row in policy["results"].values()))
            self.assertTrue(policy["sasrec_mol_gain"]["passed"])

    def test_paper_check_rejects_metric_drift_and_reversed_mol_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {"paper_verification": {"required": True, "absolute_tolerance": .01}}
            metrics = dict(zip(benchmark.METRICS, benchmark.TARGETS["sasrec"]))
            metrics["mrr"] -= .02
            with self.assertRaisesRegex(RuntimeError, "consistency check failed"):
                benchmark.verify_paper(run, "sasrec", metrics, manifest)
            manifest = {"paper_verification": {"required": True, "absolute_tolerance": .1}}
            metrics = dict(zip(benchmark.METRICS, benchmark.TARGETS["sasrec"]))
            benchmark.verify_paper(run, "sasrec", metrics, manifest)
            with self.assertRaisesRegex(RuntimeError, "consistency check failed"):
                benchmark.verify_paper(run, "sasrec_mol", metrics, manifest)

    def test_resume_preserves_completed_epochs_and_runs_queued_model(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "logs").mkdir()
            benchmark.dump(run / "manifest.json", {
                "models": {"hstu_lcar": {}, "sasrec_lcar": {}}, "commands": [], "data_sha256": {}})
            benchmark.dump(run / "source_sha256.json", {})
            original = "".join(json.dumps(epoch_row(i)) + "\n" for i in range(71))
            for name in ("hstu_lcar", "sasrec_lcar"):
                folder = run / name
                folder.mkdir()
                (folder / benchmark.MODELS[name]).write_text("# existing config\n")
            (run / "hstu_lcar/epochs.jsonl").write_text(original)
            (run / "hstu_lcar/last.pt").write_bytes(b"checkpoint")
            calls = []

            def fake_command(args, log, env, manifest):
                calls.append(args)
                config = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--gin_config_file=")))
                path = config.parent / "epochs.jsonl"
                start = 71 if config.parent.name == "hstu_lcar" else 0
                with path.open("a") as stream:
                    stream.write("".join(json.dumps(epoch_row(i)) + "\n" for i in range(start, 101)))

            with patch.object(benchmark, "command", side_effect=fake_command), patch.object(
                    benchmark, "evaluate", return_value={"metrics": epoch_row(0)["metrics"]}):
                benchmark.resume(run)
            self.assertEqual(len(calls), 2)
            self.assertIn(f"--restore_from_ckpt={run / 'hstu_lcar/last.pt'}", calls[0])
            self.assertFalse(any(arg.startswith("--restore_from_ckpt=") for arg in calls[1]))
            self.assertTrue((run / "hstu_lcar/epochs.jsonl").read_text().startswith(original))
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["resumptions"][0]["rng_state_restored"])
            self.assertIn("LCAR-SASRec", (run / "comparison.md").read_text())

    def test_resume_rejects_changed_data_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            benchmark.dump(run / "manifest.json", {"data_sha256": {"train.py": "wrong-hash"}})
            with patch.object(benchmark, "command") as command:
                with self.assertRaisesRegex(ValueError, "Dataset changed"):
                    benchmark.resume(run)
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
