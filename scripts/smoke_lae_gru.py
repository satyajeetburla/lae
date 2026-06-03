#!/usr/bin/env python3
"""Validate the paper LAE GRU config and run dummy forwards when weights exist."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "quad-swarm-rl"
DEFAULT_CONFIG = REPO_ROOT / "artifacts" / "lae" / "paper_h250_m10" / "config.json"


def resolve_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="LAE config path.")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Exit with an error if classifier/GRU checkpoint files are missing.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PACKAGE_ROOT))

    config_path = resolve_path(args.config, REPO_ROOT)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    classifier_cfg = config["classifier"]
    editor_cfg = config["editor"]
    classifier_path = resolve_path(classifier_cfg["path"], config_path.parent)
    editor_path = resolve_path(editor_cfg["path"], config_path.parent)
    missing = [path for path in (classifier_path, editor_path) if not path.exists()]
    if missing:
        print("LAE config is valid, but required checkpoint files are missing:")
        for path in missing:
            print(f"  - {path}")
        print("Copy the paper classifier and GRU LCWM weights to these paths, then rerun this script.")
        if args.require_artifacts:
            raise SystemExit(1)
        return

    import torch
    from swarm_rl.models.quad_multi_model import Classifier, LatentPredictorGRU

    classifier = Classifier(
        input_dim=int(classifier_cfg["input_dim"]),
        hidden_dims=list(classifier_cfg["hidden_dims"]),
        num_classes=int(classifier_cfg["num_classes"]),
    )
    classifier_state = torch.load(classifier_path, map_location="cpu")
    classifier_state = classifier_state.get("state_dict", classifier_state)
    classifier.load_state_dict(classifier_state, strict=True)
    classifier.eval()

    editor = LatentPredictorGRU(
        latent_dim=int(editor_cfg["latent_dim"]),
        hidden_dim=int(editor_cfg["hidden_dim"]),
        num_layers=int(editor_cfg["num_layers"]),
        dropout=float(editor_cfg["dropout"]),
    )
    editor_state = torch.load(editor_path, map_location="cpu")
    editor_state = editor_state.get("model_state", editor_state.get("state_dict", editor_state))
    editor.load_state_dict(editor_state, strict=True)
    editor.eval()

    with torch.no_grad():
        logits = classifier(torch.zeros(2, int(classifier_cfg["input_dim"])))
        pred = editor(torch.zeros(2, int(editor_cfg["window"]), int(editor_cfg["latent_dim"])))

    print(f"Loaded classifier: {classifier_path}")
    print(f"Loaded GRU LCWM: {editor_path}")
    print(f"classifier output shape={tuple(logits.shape)}")
    print(f"editor output shape={tuple(pred.shape)}")


if __name__ == "__main__":
    main()
