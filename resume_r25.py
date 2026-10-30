import json
from pathlib import Path

from core.GCN.experiments import run_study


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    manifest_path = (
        root / "experiments/sycamore_benchmark_r25/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())

    config = manifest["configuration"]
    config["action"] = "final"
    config["resume"] = True
    config["max_hours"] = None

    run_study(config, project_root=root)