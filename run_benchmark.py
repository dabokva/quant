from pathlib import Path
from benchmark_config import CONFIG
from core.GCN.experiments import run_study

if __name__ == '__main__':
    run_study(CONFIG, project_root=Path(__file__).resolve().parent)
