from pathlib import Path

from reference_comparison_config import CONFIG
from reference_comparison import run_comparison


if __name__ == "__main__":
    run_comparison(CONFIG, project_root=Path(__file__).resolve().parent)
