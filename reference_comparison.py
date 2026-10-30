from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import platform
import re
import time

import numpy as np
import scipy
from scipy.stats import beta
from scipy.special import expit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def relative_path(value):
    p = Path(value)
    require(not p.is_absolute() and p.parts and ".." not in p.parts,
            f"Expected a relative path without '..': {value}")
    return p


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.records = {}

    def sha(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = digest(path)
        return self.hashes[path]

    def bits(self, path, shots, fmt):
        path = Path(path).resolve()
        key = (path, shots, fmt)
        if key not in self.records:
            require(path.is_file(), f"Missing input: {path}")
            if fmt == "b8":
                values = np.fromfile(path, dtype=np.uint8)
                require(len(values) == shots, f"Expected {shots} bytes (one observable/shot): {path}")
                require(np.all(values <= 1), f"Nonzero extra bits: expected ONE observable in {path}")
            else:
                lines = path.read_text().splitlines()
                require(len(lines) == shots and all(x in ("0", "1") for x in lines),
                        f"Expected {shots} lines, each containing one bit: {path}")
                values = np.array([int(x) for x in lines], dtype=np.uint8)
            self.sha(path)
            self.records[key] = values
        return self.records[key]


def validate_collection(collection):
    offset = 0
    names = set()
    require(collection["samples"], "Empty acquisition collection")
    for sample in collection["samples"]:
        require(sample["name"] not in names, "Duplicate acquisition name")
        require(sample["offset"] == offset and sample["shots"] > 0,
                "Acquisition offsets must be contiguous with positive shot counts")
        require(relative_path(sample["relative_path"]).parts[0] == sample["name"],
                "Acquisition name/path mismatch")
        names.add(sample["name"])
        offset += sample["shots"]
    require(offset == collection["total_shots"], "Collection size mismatch")


def load_predictions(path, expected_ids, collection):
    with np.load(path, allow_pickle=False) as data:
        needed = ("shot_indices", "labels", "probabilities", "sample_names", "local_shot_indices")
        require(all(k in data for k in needed), f"Missing shot-alignment arrays: {path}")
        arrays = {k: data[k].copy() for k in needed}
    ids, y, p = (arrays[k] for k in needed[:3])
    require(all(a.ndim == 1 and len(a) == len(ids) for a in arrays.values()),
            f"Expected one prediction and label per shot: {path}")
    require(ids.dtype.kind in "iu" and arrays["local_shot_indices"].dtype.kind in "iu",
            "Shot identifiers must be integers")
    require(len(ids) > 0 and len(np.unique(ids)) == len(ids), "Empty or duplicate prediction IDs")
    require(np.array_equal(np.sort(ids), np.sort(expected_ids)), f"Prediction split mismatch: {path}")
    require(np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
            f"Invalid labels/probabilities: {path}")
    samples = collection["samples"]
    offsets = np.array([s["offset"] for s in samples])
    require(((ids >= 0) & (ids < collection["total_shots"])).all(), "Shot ID outside collection")
    acquisition = np.searchsorted(offsets, ids, side="right") - 1
    require(np.array_equal(arrays["sample_names"], np.array([s["name"] for s in samples])[acquisition]),
            "Saved acquisition names do not match global shot IDs")
    require(np.array_equal(arrays["local_shot_indices"], ids - offsets[acquisition]),
            "Saved local shot indices do not match global shot IDs")
    return arrays


def take_predictions(arrays, ids):
    ids = np.asarray(ids, dtype=np.int64)
    require(len(ids) > 0 and len(np.unique(ids)) == len(ids), "Invalid calibration-selection IDs")
    order = np.argsort(arrays["shot_indices"])
    position = np.searchsorted(arrays["shot_indices"][order], ids)
    require((position < len(order)).all(), "Calibration selection is missing predictions")
    positions = order[position]
    require(np.array_equal(arrays["shot_indices"][positions], ids), "Calibration selection mismatch")
    return {k: v[positions] for k, v in arrays.items()}


def align_reference(arrays, collection, data_root, reference, fmt, inputs):
    out = np.empty(len(arrays["labels"]), dtype=np.uint8)
    for sample in collection["samples"]:
        mask = arrays["sample_names"] == sample["name"]
        local = arrays["local_shot_indices"][mask]
        base = data_root / relative_path(sample["relative_path"])
        actual_path = base / "obs_flips_actual.b8"
        actual = inputs.bits(actual_path, sample["shots"], "b8")
        require(inputs.sha(actual_path) == sample["sha256"]["obs_flips_actual.b8"],
                f"Ground-truth file differs from training collection: {actual_path}")
        require(np.array_equal(actual[local], arrays["labels"][mask]),
                f"Saved labels do not match original shot order: {base}")
        predicted = inputs.bits(base / relative_path(reference), sample["shots"], fmt)
        out[mask] = predicted[local]
    return out


def calibrated_probabilities(probabilities, gate, backend="numpy"):
    p = np.asarray(probabilities, dtype=np.float32)
    require(gate["method"] in ("none", "temperature"), "Unsupported calibration method")
    temperature = float(gate["temperature"]) if gate["method"] == "temperature" else 1.0
    require(np.isfinite(temperature) and temperature > 0, "Invalid saved temperature")
    if backend == "tensorflow":
        import tensorflow as tf
        with tf.device("/CPU:0"):
            clipped = tf.clip_by_value(tf.constant(p), 1e-7, 1 - 1e-7)
            return tf.sigmoid((tf.math.log(clipped) - tf.math.log1p(-clipped)) /
                              tf.constant(temperature, tf.float32)).numpy()
    clipped = np.clip(p, np.float32(1e-7), np.float32(1 - 1e-7))
    logits = (np.log(clipped) - np.log1p(-clipped)) / np.float32(temperature)
    return expit(logits).astype(np.float32)


def paired_error_interval(harm, benefit, n, tail):
    def bounds(k):
        low = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
        high = 1.0 if k == n else float(beta.isf(tail, k + 1, n - k))
        return low, high
    hl, hu = bounds(harm)
    bl, bu = bounds(benefit)
    return hl - bu, hu - bl


def compare_policies(labels, probabilities, reference, gate, thresholds, tail, bounds=True, backend="numpy"):
    y = np.asarray(labels)
    raw_wrong = (np.asarray(probabilities) > 0.5) != y
    q = calibrated_probabilities(probabilities, gate, backend)
    nn_wrong = (q > 0.5) != y
    ref_wrong = np.asarray(reference) != y
    confidence = np.maximum(q, np.float32(1) - q)
    n = len(y)
    require(n > 0, "Cannot evaluate an empty split")
    order = np.argsort(-confidence, kind="stable")
    sorted_conf = confidence[order]
    prefix = {}
    for name, values in {"neural_errors": nn_wrong, "reference_errors": ref_wrong,
                         "introduced_errors": nn_wrong & ~ref_wrong,
                         "corrected_errors": ~nn_wrong & ref_wrong}.items():
        prefix[name] = np.r_[0, np.cumsum(values[order], dtype=np.int64)]
    policies = [("delegate_all", None, False),
                ("frozen_gate", gate["threshold"] if gate["enabled"] else None, bool(gate["enabled"]))]
    policies += [("threshold", float(t), True) for t in thresholds]
    result = []
    for policy, threshold, enabled in policies:
        if enabled:
            require(threshold is not None and 0.5 <= threshold <= 1, "Invalid gate threshold")
        accepted = int(np.searchsorted(-sorted_conf, -np.float32(threshold), side="right")) if enabled else 0
        values = {key: int(value[accepted]) for key, value in prefix.items()}
        harm, benefit = values["introduced_errors"], values["corrected_errors"]
        hybrid_errors = int(ref_wrong.sum()) + harm - benefit
        lower = upper = None
        if bounds:
            lower, upper = paired_error_interval(harm, benefit, n, tail) if enabled else (0.0, 0.0)
        result.append({
            "policy": policy, "threshold": threshold, "enabled": enabled,
            "shots": n, "accepted_shots": accepted, "delegated_shots": n - accepted,
            "coverage": accepted / n, "reference_call_reduction": accepted / n,
            "neural_accuracy": float(1 - raw_wrong.mean()),
            "calibrated_neural_accuracy": float(1 - nn_wrong.mean()),
            "reference_accuracy": float(1 - ref_wrong.mean()),
            "accepted_neural_accuracy": 1 - values["neural_errors"] / accepted if accepted else None,
            "accepted_reference_accuracy": 1 - values["reference_errors"] / accepted if accepted else None,
            "accepted_neural_errors": values["neural_errors"],
            "accepted_reference_errors": values["reference_errors"],
            "introduced_errors": harm, "corrected_errors": benefit, "net_errors_removed": benefit - harm,
            "hybrid_errors": hybrid_errors, "hybrid_accuracy": 1 - hybrid_errors / n,
            "accuracy_change_pp": 100 * (benefit - harm) / n,
            "error_difference": (harm - benefit) / n,
            "error_difference_lower": lower, "error_difference_upper": upper,
        })
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def escaped(text):
    return "".join({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
                    "{": r"\{", "}": r"\}", "$": r"\$", "\\": r"\textbackslash{}"}.get(x, x) for x in str(text))


def policy_name(row):
    if row["policy"] == "threshold":
        return f'{row["threshold"]:g}'
    return "Delegate all" if row["policy"] == "delegate_all" else "Original gate"


def save_table(path, title, rows):
    def percent(x):
        return "--" if x is None else f"{100*x:.2f}"
    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + escaped(title) + r". Accuracies and acceptance are percentages; change is in percentage points relative to the classical decoder.}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Threshold & Accepted & Neural (acc.) & Classical (acc.) & Hybrid & Change \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join([escaped(policy_name(row)), percent(row["coverage"]),
                                  percent(row["accepted_neural_accuracy"]), percent(row["accepted_reference_accuracy"]),
                                  percent(row["hybrid_accuracy"]), f'{row["accuracy_change_pp"]:+.3f}']) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\footnotesize Neural (acc.) and Classical (acc.) refer to the same accepted subset. An empty accepted subset has undefined accuracy.", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def save_plot(path, title, rows, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(rows))
    labels = [policy_name(row) for row in rows]
    def values(key, scale=100):
        return np.array([np.nan if r[key] is None else r[key] * scale for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, layout="constrained")
    ax = axes[0, 0]
    ax.plot(x, values("hybrid_accuracy"), "o-", color="#197650", label="Hybrid")
    ax.axhline(values("reference_accuracy")[0], color="#325b91", ls="--", label="Classical alone")
    ax.axhline(values("neural_accuracy")[0], color="#a36017", ls=":", label="Neural alone")
    ax.set(title="Accuracy over all shots", ylabel="Accuracy (%)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.bar(x, values("coverage"), color="#765390")
    ax.set(title="Acceptance and reference-call reduction", ylabel="Shots accepted (%)", ylim=(0, 103))
    ax = axes[1, 0]
    ax.plot(x, values("accepted_neural_accuracy"), "o-", color="#a36017", label="Neural on accepted shots")
    ax.plot(x, values("accepted_reference_accuracy"), "s--", color="#325b91", label="Classical on accepted shots")
    ax.set(title="Accuracy on the SAME accepted subset", ylabel="Accuracy (%)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    gain = values("accuracy_change_pp", 1)
    lo, hi = -values("error_difference_upper"), -values("error_difference_lower")
    ax.errorbar(x, gain, yerr=[np.maximum(0, gain-lo), np.maximum(0, hi-gain)],
                fmt="o", color="#197650", capsize=3)
    ax.axhline(0, color="#325b91", ls="--")
    ax.set(title=f"Hybrid accuracy change; simultaneous {100*(1-alpha):g}% bounds",
           ylabel="Accuracy change (percentage points)")
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_xlabel("Confidence threshold / fixed policy")
    fig.suptitle(title + "\nOriginal gate shown separately; configured thresholds bypass its enable/disable decision.", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def run_comparison(config, project_root=None):
    start = time.monotonic()
    root = Path(project_root or Path(__file__).resolve().parent).resolve()
    def resolve(value):
        p = Path(value).expanduser()
        return (p if p.is_absolute() else root / p).resolve()
    thresholds = [float(t) for t in config["thresholds"]]
    require(thresholds and all(np.isfinite(t) and .5 <= t <= 1 for t in thresholds)
            and len(set(thresholds)) == len(thresholds), "Thresholds must be unique numbers in [0.5, 1]")
    thresholds.sort()
    require(0 < config["alpha"] < 1, "alpha must be between zero and one")
    require(config["reference_format"] in ("b8", "01"), "reference_format must be b8 or 01")
    backend = config.get("calibration_backend", "numpy")
    require(backend in ("numpy", "tensorflow"), "calibration_backend must be numpy or tensorflow")
    if backend == "tensorflow":
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    require(config["references"] and config["studies"], "Configure at least one study and reference")
    for name in [*config["references"], *config["studies"]]:
        require(re.fullmatch(r"[A-Za-z0-9_-]+", name), f"Invalid study/reference name: {name}")
    for path in config["references"].values():
        relative_path(path)
    wanted = config.get("models")
    require(wanted is None or (isinstance(wanted, list) and wanted and len(set(wanted)) == len(wanted)),
            "models must be None or a nonempty list of unique run keys")
    inputs, jobs = Inputs(), []
    data_root, output_root = resolve(config["data_root"]), resolve(config["output_root"])
    sources = [data_root]
    for tag, location in config["studies"].items():
        study = resolve(location)
        sources.append(study)
        manifest = read_json(study / "manifest.json")
        frozen = read_json(study / "final_selection.json")
        require(manifest["schema_version"] == 3 and manifest["status"] == "complete",
                f"Study must be a completed schema-3 benchmark: {study}")
        require(frozen["configuration_sha256"] == manifest["configuration_sha256"], "Frozen configuration mismatch")
        eligible = {k: v for k, v in manifest["runs"].items() if v["eligible"]}
        require(eligible, f"No eligible models: {study}")
        require(wanted is None or set(wanted) <= eligible.keys(), f"Configured model missing or rejected in {study}")
        inputs.sha(study / "manifest.json")
        inputs.sha(study / "final_selection.json")
        for key, entry in eligible.items():
            if wanted is not None and key not in wanted:
                continue
            relative_path(key)
            require(entry["development_complete"], f"Incomplete development: {key}")
            relative = Path(entry["path"]).relative_to(manifest["configuration"]["output_root"])
            run = study / relative_path(relative)
            evaluation = read_json(run / "evaluation_test.json")
            gate = read_json(run / "gate.json")
            require(evaluation["subset"] == "test", "Expected a held-out test report")
            identity = frozen["identities"][key]
            require(identity == evaluation["run_identity"] == gate["run_identity"], f"Model/gate identity mismatch: {key}")
            require(inputs.sha(run / "gate.json") == evaluation["gate_sha256"] == frozen["gate_hashes"][key],
                    f"Changed frozen gate: {key}")
            for name, expected in identity.items():
                if name == "precision":
                    continue
                path = run / relative_path(name)
                if config["verify_artifact_hashes"] or name in ("collection.json", "shot_indices.npz"):
                    require(inputs.sha(path) == expected, f"Changed frozen artifact: {path}")
            collection = read_json(run / "collection.json")
            validate_collection(collection)
            require(collection["configuration"] == manifest["configuration"]["configuration"], "Round/configuration mismatch")
            for sample in collection["samples"]:
                require(Path(sample["relative_path"]) == Path(sample["name"]) / collection["configuration"],
                        "Acquisition configuration differs from study")
                base = data_root / relative_path(sample["relative_path"])
                for filename in ["obs_flips_actual.b8", *config["references"].values()]:
                    require((base / relative_path(filename)).is_file(),
                            f"Missing input: {base / filename}\nSet data_root to the directory containing sample_00, sample_01, ...")
            jobs.append((tag, key, run, collection, gate, evaluation))
    for source in sources:
        require(not output_root.is_relative_to(source) and not source.is_relative_to(output_root),
                "output_root must be separate from input studies and data")
    comparisons = 2 * len(config["references"]) * sum(
        len(thresholds) + len(g["calibration_curve"]) + 2 for _, _, _, _, g, _ in jobs)
    tail = config["alpha"] / (4 * comparisons)
    rows, per_sample, groups, calibration_replay = [], [], [], []
    membership = {}
    for index, (tag, key, run, collection, gate, evaluation) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {tag} / {key}: align saved shots and compare thresholds", flush=True)
        with np.load(run / "shot_indices.npz", allow_pickle=False) as archive:
            parts = {k: archive[k].copy() for k in ("train", "validation", "calibration", "test")}
        names = list(parts)
        for i, name in enumerate(names):
            require(len(np.unique(parts[name])) == len(parts[name]), "Duplicate partition IDs")
            for other in names[i+1:]:
                require(not np.intersect1d(parts[name], parts[other]).size, "Training/evaluation partition overlap")
        fit, selection = (np.asarray(gate[k], dtype=np.int64) for k in ("fit_shot_indices", "selection_shot_indices"))
        require(len(np.unique(fit)) == len(fit) and not np.intersect1d(fit, selection).size,
                "Calibration fit/selection overlap or duplicates")
        require(np.array_equal(np.sort(np.r_[fit, selection]), np.sort(parts["calibration"])),
                "Gate calibration halves do not match the reserved calibration partition")
        calibration_file = run / "calibration_predictions.npz"
        test_file = run / relative_path(evaluation["prediction_file"])
        calibration = take_predictions(load_predictions(calibration_file, parts["calibration"], collection), selection)
        test = load_predictions(test_file, parts["test"], collection)
        cq = calibrated_probabilities(calibration["probabilities"], gate, backend)
        confidence = np.maximum(cq, np.float32(1) - cq)
        differences = []
        for original in gate["calibration_curve"]:
            mask = confidence >= np.float32(original["threshold"])
            accepted, incorrect = int(mask.sum()), int(np.sum((cq[mask] > .5) != calibration["labels"][mask]))
            if accepted != original["accepted"] or incorrect != original["incorrect"]:
                differences.append({"threshold": original["threshold"],
                                    "saved_accepted": original["accepted"], "replayed_accepted": accepted,
                                    "saved_incorrect": original["incorrect"], "replayed_incorrect": incorrect})
        calibration_replay.append({"study": tag, "model": key, "differences": differences})
        if differences:
            print(f"  Calibration replay: {len(differences)} cutoffs differ from the gate-fitting pass; "
                  "saved-prediction results and discrepancies are retained separately.", flush=True)
        for path in (calibration_file, test_file, run / "evaluation_test.json"):
            inputs.sha(path)
        require(int(np.sum((test["probabilities"] > .5) != test["labels"])) == evaluation["results"]["neural"]["incorrect"],
                "Saved test predictions do not reproduce the reported neural error count")
        ordered = np.argsort(test["shot_indices"])
        fingerprint = hashlib.sha256(test["shot_indices"][ordered].tobytes() + test["labels"][ordered].tobytes()).hexdigest()
        require(tag not in membership or membership[tag] == fingerprint, "Models use different test records within a study")
        membership[tag] = fingerprint
        for split, arrays in (("calibration_selection", calibration), ("test", test)):
            for reference, filename in config["references"].items():
                ref = align_reference(arrays, collection, data_root, filename, config["reference_format"], inputs)
                values = compare_policies(arrays["labels"], arrays["probabilities"], ref, gate, thresholds, tail, backend=backend)
                if split == "test":
                    require(values[1]["accepted_shots"] == evaluation["gate"]["accepted"]["shots"],
                            "Frozen TEST gate coverage does not reproduce the original report. "
                            "Try calibration_backend='tensorflow' in the benchmark environment.")
                    require(values[1]["accepted_neural_errors"] == evaluation["gate"]["accepted"]["incorrect"],
                            "Frozen gate errors do not reproduce the original test report")
                context = {"study": tag, "model": key, "reference": reference, "subset": split}
                rows.extend({**context, **row} for row in values)
                groups.append({**context, "temperature": gate["temperature"], "rows": values})
                for sample in collection["samples"]:
                    mask = arrays["sample_names"] == sample["name"]
                    if not mask.any():
                        continue
                    local = compare_policies(arrays["labels"][mask], arrays["probabilities"][mask],
                                             ref[mask], gate, thresholds, tail, bounds=False, backend=backend)
                    per_sample.extend({**context, "sample": sample["name"], **row} for row in local)
    output = output_root / datetime.now(timezone.utc).strftime("comparison_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "per_acquisition.csv", per_sample)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "confidence": "max(q,1-q), saved temperature; unrounded probabilities",
        "threshold_selection": "None: all configured thresholds are evaluated as fixed exploratory policies",
        "interval_method": "Clopper-Pearson marginal discordance bounds with a Bonferroni union bound",
        "simultaneous_comparisons": comparisons, "binomial_tail_probability": tail,
        "interval_assumption": "Independent shots; bounds do not account for within-acquisition dependence or distribution shift",
        "scope": "One complete experimental shot; workload reduction means avoided reference-decoder calls, not execution time",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "inputs_sha256": {str(p): h for p, h in inputs.hashes.items()},
        "calibration_replay": calibration_replay, "comparisons": groups,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for index, group in enumerate(groups, 1):
        relative = Path(group["study"]) / group["model"] / f'{group["reference"]}_{group["subset"]}'
        title = f'{group["study"]}: {group["model"]}; {group["reference"]}; {group["subset"].replace("_", " ")}'
        save_table((output / "tables" / relative).with_suffix(".tex"), title, group["rows"])
        if config["plots"]:
            print(f"  Plot {index}/{len(groups)}: {relative}", flush=True)
            save_plot(output / "plots" / relative, title, group["rows"], config["alpha"])
    (output / "README.txt").write_text(
        "Fixed-threshold reference comparison\n\n"
        "comparison.csv: pooled results; per_acquisition.csv: acquisition-specific results.\n"
        "plots/: static PDF and PNG figures. tables/: LaTeX tables (requires booktabs).\n"
        "comparison.json: methods, settings, numerical results, and input hashes.\n\n"
        "Calibration-selection rows exclude the temperature-fitting half. Test rows are held-out shot comparisons.\n"
        "Gate fitting used a separate inference pass. Its differences from saved calibration predictions are in calibration_replay.\n"
        "All configured thresholds are reported; the script does not select a winner using test results.\n"
        "This is a follow-up exploratory analysis of existing studies. Keep it distinct from the original 1% gate.\n"
        "Original gate respects its saved enabled flag. Fixed thresholds override it for this analysis only.\n"
        "Corrected errors: classical wrong, neural correct on accepted shots. Introduced errors: the reverse.\n"
        "Hybrid error = classical error + introduced errors - corrected errors.\n"
        "Error-difference bounds cover all reported global comparisons, assuming independent shots.\n"
        "They do not guarantee future accuracy or account for correlation between shots.\n"
        "An empty accepted set has undefined accepted accuracy. Calls avoided are not measured time saved.\n"
        "There is no training, new neural inference, gate fitting, or latency measurement in this script.\n")
    print(f"Complete in {time.monotonic()-start:.1f}s: {output}", flush=True)
    return output
