"""Training loop entrypoint (stage 3).

Not implemented yet: this stub defines the intended entrypoint shape
(config-driven, GPU-first) without running any training.
"""

from __future__ import annotations

from pathlib import Path


def train(config_path: Path | str) -> None:
    """Run a training job from a YAML config (e.g. ``configs/baseline.yaml``).

    Args:
        config_path: path to a YAML experiment config.
    """
    raise NotImplementedError("Stage 3: training loop not yet implemented.")


if __name__ == "__main__":
    raise SystemExit("Training is not implemented yet; see train() docstring.")
