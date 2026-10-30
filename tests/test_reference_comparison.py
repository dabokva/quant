
from copy import deepcopy
from pathlib import Path
import csv
import json
import tempfile
import unittest

import numpy as np

from reference_comparison import (Inputs, align_reference, calibrated_probabilities,
                                  compare_policies, digest, load_predictions,
                                  paired_error_interval, run_comparison)


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    study, data = root / "study", root / "data"
    key = "fixed/batch_64/baseline"
    relative = key + "/run"
    run = study / relative
    run.mkdir(parents=True)
    configuration = "d5_at_q6_5/X/r05"
    y = np.tile(np.array([0, 1], dtype=np.uint8), 16)
    reference = y.copy()
    reference[[9, 14, 25, 28, 31]] ^= 1
    p = np.where(y, .9, .1).astype(np.float32)
    p[[10, 12, 14, 26, 30]] = 1 - p[[10, 12, 14, 26, 30]]
    p[[11, 27, 29]] = .51
    samples = []
    for i in range(2):
        name = f"sample_{i:02d}"
        folder = data / name / configuration
        folder.mkdir(parents=True)
        actual = folder / "obs_flips_actual.b8"
        y[i*16:(i+1)*16].tofile(actual)
        refs = folder / "obs_flips_predicted/dem_correlations"
        refs.mkdir(parents=True)
        reference[i*16:(i+1)*16].tofile(refs / "correlated_matching.b8")
        samples.append({"name": name, "relative_path": f"{name}/{configuration}",
                        "offset": i*16, "shots": 16,
                        "sha256": {"obs_flips_actual.b8": digest(actual)}})
    collection = {"configuration": configuration, "total_shots": 32, "samples": samples}
    write_json(run / "collection.json", collection)
    parts = {"train": np.r_[0:4, 16:20], "validation": np.r_[4:6, 20:22],
             "calibration": np.r_[6:12, 22:28], "test": np.r_[12:16, 28:32]}
    np.savez(run / "shot_indices.npz", **parts)
    def save(name, ids):
        ids = np.asarray(ids)[::-1]
        np.savez(run / name, shot_indices=ids, labels=y[ids], probabilities=p[ids],
                 sample_names=np.where(ids < 16, "sample_00", "sample_01"), local_shot_indices=ids % 16)
    save("calibration_predictions.npz", parts["calibration"])
    save("evaluation_test.predictions.npz", parts["test"])
    (run / "weights.npz").write_bytes(b"synthetic model identity; never loaded")
    identity = {name: digest(run / name) for name in ("collection.json", "shot_indices.npz", "weights.npz")}
    identity["precision"] = "float32"
    fit, selection = np.r_[6:9, 22:25], np.r_[9:12, 25:28]
    gate = {"enabled": False, "threshold": None, "method": "temperature", "temperature": 1.0,
            "fit_shot_indices": fit.tolist(), "selection_shot_indices": selection.tolist(),
            "run_identity": identity, "calibration_curve": []}
    q = calibrated_probabilities(p[selection], gate)
    for t in np.linspace(.5, 1, 101, dtype=np.float32):
        mask = np.maximum(q, 1-q) >= t
        gate["calibration_curve"].append({"threshold": float(t), "accepted": int(mask.sum()),
                                           "incorrect": int(np.sum((q[mask]>.5) != y[selection][mask]))})
    write_json(run / "gate.json", gate)
    evaluation = {"subset": "test", "run_identity": identity, "gate_sha256": digest(run / "gate.json"),
                  "prediction_file": "evaluation_test.predictions.npz",
                  "results": {"neural": {"incorrect": int(np.sum((p[parts['test']]>.5) != y[parts['test']]))}},
                  "gate": {"accepted": {"shots": 0, "incorrect": 0}}}
    write_json(run / "evaluation_test.json", evaluation)
    manifest = {"schema_version": 3, "status": "complete", "configuration_sha256": "synthetic",
                "configuration": {"configuration": configuration, "output_root": "/original/work/study"},
                "runs": {key: {"eligible": True, "development_complete": True,
                               "path": "/original/work/study/" + relative}}}
    write_json(study / "manifest.json", manifest)
    write_json(study / "final_selection.json", {"configuration_sha256": "synthetic",
               "identities": {key: identity}, "gate_hashes": {key: digest(run / "gate.json")}})
    config = {"data_root": str(data), "studies": {"r05": str(study)}, "output_root": str(root / "output"),
              "references": {"correlated_matching": "obs_flips_predicted/dem_correlations/correlated_matching.b8"},
              "reference_format": "b8", "models": None, "thresholds": [.5, .8, .95, 1.0],
              "alpha": .05, "calibration_backend": "numpy", "verify_artifact_hashes": True, "plots": False}
    return config, run, parts, collection, reference


class ComparisonTests(unittest.TestCase):
    def test_error_exchange_example(self):
        y = np.zeros(1268, dtype=np.uint8)
        p = np.full(1268, .001, dtype=np.float32)
        p[:21] = .999
        gate = {"enabled": False, "threshold": None, "temperature": 1, "method": "none"}
        for reference_errors, net in [(30, 9), (5, -16)]:
            ref = y.copy()
            ref[:reference_errors] = 1
            rows = compare_policies(y, p, ref, gate, [.985], .001)
            self.assertEqual(rows[2]["accepted_shots"], 1268)
            self.assertEqual(rows[2]["net_errors_removed"], net)
            self.assertEqual(rows[2]["hybrid_errors"], 21)
            self.assertEqual(rows[1]["accepted_shots"], 0)
            self.assertIsNone(rows[1]["accepted_neural_accuracy"])
            self.assertEqual(rows[1]["error_difference_upper"], 0)

    def test_identical_subset_matters(self):
        y = np.zeros(100, dtype=np.uint8)
        p = np.r_[np.full(3, .99), np.full(7, .01), np.full(90, .49)]
        ref = y.copy()
        ref[10:60] = 1 
        gate = {"enabled": False, "threshold": None, "temperature": 1, "method": "none"}
        row = compare_policies(y, p, ref, gate, [.9], .001)[2]
        self.assertEqual(row["accepted_reference_accuracy"], 1)
        self.assertAlmostEqual(row["accepted_neural_accuracy"], .7)
        self.assertEqual(row["introduced_errors"], 3)
        self.assertAlmostEqual(row["accuracy_change_pp"], -3)

    def test_b8_records_are_byte_padded_per_shot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "one.b8"
            path.write_bytes(bytes([0, 1, 1, 0, 1]))
            np.testing.assert_array_equal(Inputs().bits(path, 5, "b8"), [0, 1, 1, 0, 1])
            with self.assertRaisesRegex(ValueError, "Expected 8 bytes"):
                Inputs().bits(path, 8, "b8")
            path.write_bytes(bytes([2]))
            with self.assertRaisesRegex(ValueError, "ONE observable"):
                Inputs().bits(path, 1, "b8")

    def test_threshold_and_pair_bounds(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([.1, .9, .5, .5], dtype=np.float32)
        gate = {"enabled": True, "threshold": .9, "temperature": 1, "method": "none"}
        rows = compare_policies(y, p, y, gate, [.5, .9, 1], .001)
        self.assertEqual(rows[1]["accepted_shots"], rows[3]["accepted_shots"])
        self.assertEqual(rows[2]["accepted_shots"], 4)
        self.assertEqual(rows[4]["accepted_shots"], 0)
        self.assertGreater(rows[4]["error_difference_upper"], 0)
        for h, b in [(0, 0), (5, 0), (0, 5), (3, 8), (100, 0)]:
            low, high = paired_error_interval(h, b, 100, .001)
            self.assertLessEqual(low, (h-b)/100)
            self.assertGreaterEqual(high, (h-b)/100)

    def test_end_to_end_outputs_and_unchanged_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, run, parts, collection, reference = fixture(root)
            before = {p: digest(p) for p in root.rglob("*") if p.is_file()}
            arrays = load_predictions(run / "evaluation_test.predictions.npz", parts["test"], collection)
            aligned = align_reference(arrays, collection, Path(config["data_root"]),
                                      config["references"]["correlated_matching"], "b8", Inputs())
            np.testing.assert_array_equal(aligned, reference[arrays["shot_indices"]])
            config["plots"] = True
            output = run_comparison(config, root)
            report = read_json(output / "comparison.json")
            self.assertEqual(len(report["comparisons"]), 2)
            self.assertEqual(len(list((output / "plots").rglob("*.pdf"))), 2)
            with (output / "comparison.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual({p: digest(p) for p in before}, before)
            for group in report["comparisons"]:
                for row in group["rows"]:
                    self.assertEqual(row["accepted_shots"] + row["delegated_shots"], row["shots"])
                    self.assertAlmostEqual(row["hybrid_accuracy"] - row["reference_accuracy"],
                                           row["accuracy_change_pp"] / 100)

    def test_wrong_ground_truth_fails_without_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, *_ = fixture(root)
            actual = next((root / "data").rglob("obs_flips_actual.b8"))
            values = bytearray(actual.read_bytes())
            values[0] ^= 1
            actual.write_bytes(values)
            with self.assertRaisesRegex(ValueError, "Ground-truth file differs"):
                run_comparison(config, root)
            self.assertFalse((root / "output").exists())

    def test_calibration_replay_drift_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, run, *_ = fixture(root)
            gate = read_json(run / "gate.json")
            gate["calibration_curve"][0]["accepted"] -= 1
            write_json(run / "gate.json", gate)
            evaluation = read_json(run / "evaluation_test.json")
            evaluation["gate_sha256"] = digest(run / "gate.json")
            write_json(run / "evaluation_test.json", evaluation)
            frozen = read_json(root / "study/final_selection.json")
            frozen["gate_hashes"]["fixed/batch_64/baseline"] = digest(run / "gate.json")
            write_json(root / "study/final_selection.json", frozen)
            output = run_comparison(config, root)
            report = read_json(output / "comparison.json")
            self.assertEqual(len(report["calibration_replay"][0]["differences"]), 1)
            self.assertEqual(report["calibration_replay"][0]["differences"][0]["replayed_accepted"], 6)

    def test_multiple_references_and_acquisition_totals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, *_ = fixture(root)
            config["references"]["second_reference"] = config["references"]["correlated_matching"]
            output = run_comparison(config, root)
            with (output / "comparison.csv").open() as stream:
                pooled = list(csv.DictReader(stream))
            with (output / "per_acquisition.csv").open() as stream:
                local = list(csv.DictReader(stream))
            self.assertEqual(len(pooled), 24)
            for row in pooled:
                matching = [r for r in local if all(r[k] == row[k] for k in
                            ("study", "model", "reference", "subset", "policy", "threshold"))]
                self.assertEqual(len(matching), 2)
                for name in ("shots", "accepted_shots", "introduced_errors", "corrected_errors", "hybrid_errors"):
                    self.assertEqual(sum(int(r[name]) for r in matching), int(row[name]))

    def test_tampered_local_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, run, parts, collection, _ = fixture(root)
            path = run / "evaluation_test.predictions.npz"
            with np.load(path) as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays["local_shot_indices"][0] ^= 1
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "local shot indices"):
                load_predictions(path, parts["test"], collection)

    def test_missing_reference_and_incomplete_study_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, *_ = fixture(root)
            wrong = deepcopy(config)
            wrong["references"]["correlated_matching"] = "missing.b8"
            with self.assertRaisesRegex(ValueError, "Missing input"):
                run_comparison(wrong, root)
            path = root / "study/manifest.json"
            m = read_json(path)
            m["status"] = "interrupted"
            write_json(path, m)
            with self.assertRaisesRegex(ValueError, "completed schema-3"):
                run_comparison(config, root)
            self.assertFalse((root / "output").exists())


def read_json(path):
    return json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
