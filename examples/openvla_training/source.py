"""Explicitly download the pinned model definition; no checkpoint weights."""

import os
from pathlib import Path

MODEL_ID = "openvla/openvla-7b"
REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
DEFAULT_SNAPSHOT = Path(
    os.environ.get(
        "FHN_OPENVLA_SNAPSHOT",
        Path(__file__).resolve().parents[2] / "build/openvla-model",
    )
)


def main():
    import argparse
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_SNAPSHOT))
    args = parser.parse_args()
    path = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        local_dir=args.output,
        allow_patterns=[
            "config.json",
            "configuration_prismatic.py",
            "modeling_prismatic.py",
        ],
    )
    print(path)


if __name__ == "__main__":
    main()
