"""flwr run installs the app under ~/.flwr/apps; features/ and runs/ stay in the project directory.
Without touching contract.py, only the path constants are redirected at runtime."""
from pathlib import Path

import contract


def apply_project_dir(run_config) -> None:
    root = str(run_config.get("project-dir", "") or "")
    if root:
        contract.FEATURES_DIR = Path(root) / "features"
        contract.RUNS_DIR = Path(root) / "runs"
