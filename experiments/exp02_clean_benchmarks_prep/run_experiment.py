"""Inspect the current input files without cleaning or relabeling them."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.visnotes_common import main


if __name__ == "__main__":
    sys.exit(main("prep"))
